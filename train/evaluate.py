import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 프로젝트 루트 경로를 파이썬 모듈 검색 경로에 등록합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig
from train.utils import NumpyDataset

def evaluate(args):
    print("=" * 60)
    print("📊 Dense Transformer — Model Evaluation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # [1단계] 검증용 모델 인스턴스 객체 생성
    # -------------------------------------------------------------------------
    print("Instantiating model...")
    config = DenseTransformerConfig(
        vocab_size=32000,
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=3072,
        max_seq_len=args.block_size,
        dropout=0.0  # 평가 시에는 드롭아웃 비활성화
    )
    model = DenseTransformer(config)
    
    # -------------------------------------------------------------------------
    # [2단계] 디스크로부터 최근 가중치 체크포인트(.pt) 자동 스캔 및 로드
    # -------------------------------------------------------------------------
    ckpts = glob.glob(os.path.join(args.ckpt_dir, f"{args.checkpoint_pattern}*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"'{args.checkpoint_pattern}' 패턴과 매치되는 체크포인트를 {args.ckpt_dir} 에서 찾을 수 없습니다.")
        
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_ckpt}...")
    
    # PyTorch 2.6+ 환경 대응을 위해 weights_only=False 옵션을 필수적으로 추가합니다.
    checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    
    # 만일 Accelerate/DDP 분산 훈련으로 인해 가중치 텐서명에 "module." 래핑 접두사가 붙어있다면,
    # 순수 단일 CPU/MPS 모델에 주입하기 위해 접두사 문자열("module.")을 사전에 발라내는 처리를 합니다.
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    step = checkpoint.get('step', -1)
    print(f"Loaded checkpoint at step {step}")
    
    # 가속 연산 디바이스 선택 (Mac: mps, NVIDIA GPU: cuda, 기타 디폴트: cpu)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running evaluation on: {device}")
    model.to(device)
    # 모델을 평가 모드(evaluation)로 전환하여 dropout 등 훈련용 연산을 비활성화합니다.
    model.eval()
    
    # -------------------------------------------------------------------------
    # [3단계] 검증(Val) 데이터셋 준비 및 데이터로더 설정
    # -------------------------------------------------------------------------
    val_npy = os.path.join(args.data_dir, "val.npy")
    if not os.path.exists(val_npy):
        raise FileNotFoundError(f"Validation dataset not found at {val_npy}. Please run prepare_data.py first.")
        
    dataset = NumpyDataset(val_npy)
    batch_size = 2 if args.smoke_test else args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # -------------------------------------------------------------------------
    # [4단계] 정방향 검증 손실(Loss) 연산 루프
    # -------------------------------------------------------------------------
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    
    print("Evaluating model...")
    # 역전파 그래디언트 그래프를 그리지 않아 연산 메모리 소모를 80% 이상 줄여줍니다.
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # 순전파
            logits, loss, main_loss = model(input_ids, labels)
            
            # shift_logits의 특성상 마지막 1개 토큰 예측은 타겟이 없으므로, (T - 1) 개의 토큰에 크로스엔트로피 손실을 누적합니다.
            num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
            total_loss += main_loss.item() * num_tokens
            total_tokens += num_tokens
            total_batches += 1
            
            # 스모크 테스트의 경우 5개 배치만 빠른 검증하고 마칩니다.
            if args.smoke_test and total_batches >= 5:
                break
        
    # -------------------------------------------------------------------------
    # [6단계] Validation Perplexity (PPL) 최종 도출
    # -------------------------------------------------------------------------
    # PPL = exp(평균 크로스엔트로피 손실)
    # 모델이 다음 단어를 예측할 때 평균적으로 몇 개의 단어 후보 사이에서 헷갈려 하는지를 나타내는 수치입니다 (낮을수록 우수).
    mean_loss = total_loss / max(1, total_tokens)
    val_ppl = np.exp(mean_loss)
    print(f"  Validation Loss: {mean_loss:.4f}")
    print(f"  Validation PPL:  {val_ppl:.4f}")
    
    # -------------------------------------------------------------------------
    # [7단계] 자동화 평가 결과 JSON 리포트 디스크 영구 저장
    # -------------------------------------------------------------------------
    # PPL 50 미만이어야 PASS로 판정합니다.
    report = {
        "step": step,
        "validation_cross_entropy": mean_loss,
        "validation_perplexity": val_ppl,
        "status": "pass" if (val_ppl < 50.0) else "fail"
    }
    
    if args.smoke_test:
        report["status"] = "pass_smoke_test"
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport successfully saved to: {args.output_file}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="drive_mock/checkpoints")
    parser.add_argument("--checkpoint_pattern", type=str, default="dense_")
    parser.add_argument("--data_dir", type=str, default="train/data")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="drive_mock/reports/evaluation_report.json")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    evaluate(args)

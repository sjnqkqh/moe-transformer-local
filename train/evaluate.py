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

from model.moe_transformer import MoETransformer
from model.moe_layer import MoETransformerBlock

class NumpyDataset(Dataset):
    def __init__(self, npy_path: str):
        # mmap_mode="r"을 활성화하여 필요한 인덱스의 데이터 블록만 디스크에서 즉석 로드합니다.
        self.data = np.load(npy_path, mmap_mode="r")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # inputs와 labels 모두 동일 시퀀스로 피딩합니다.
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

def evaluate(args):
    print("=" * 60)
    print("📊 MoE Transformer — Model Evaluation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # [1단계] 검증용 모델 인스턴스 객체 생성
    # -------------------------------------------------------------------------
    print("Instantiating model...")
    model = MoETransformer(
        vocab_size=32000,
        d_model=768,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        num_experts=4,
        k=2,
        max_seq_len=args.block_size
    )
    
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
    # [4단계] 라우팅 상태 수집용 포워드 훅(Hook)을 MoE router에 등록
    # -------------------------------------------------------------------------
    selected_indices = []
    def hook_fn(module, input, output):
        # output[1] = top_k_indices (S, k)
        selected_indices.append(output[1].detach().cpu())
        
    hooks = []
    for layer in model.layers:
        if isinstance(layer, MoETransformerBlock):
            hooks.append(layer.ffn.router.register_forward_hook(hook_fn))
            
    # -------------------------------------------------------------------------
    # [5단계] 정방향 검증 손실(Loss) 연산 루프
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
            logits, loss, main_loss, aux_loss, z_loss = model(input_ids, labels)
            
            # shift_logits의 특성상 마지막 1개 토큰 예측은 타겟이 없으므로, (T - 1) 개의 토큰에 크로스엔트로피 손실을 누적합니다.
            num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
            total_loss += main_loss.item() * num_tokens
            total_tokens += num_tokens
            total_batches += 1
            
            # 스모크 테스트의 경우 5개 배치만 빠른 검증하고 마칩니다.
            if args.smoke_test and total_batches >= 5:
                break
                
    # 훅 제거를 통한 메모리 관리
    for h in hooks:
        h.remove()
        
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
    # [7단계] 전문가 균등 할당(Load Balancing CV) 지표 연산
    # -------------------------------------------------------------------------
    # 수집한 전문가 할당 기록 병합
    all_indices = torch.cat(selected_indices, dim=0)
    expert_counts = torch.zeros(4)
    for idx in all_indices.view(-1):
        if idx.item() < 4:
            expert_counts[idx.item()] += 1
            
    total_selections = expert_counts.sum().item()
    expert_percentages = (expert_counts / total_selections).tolist()
    
    # Coefficient of Variation (CV) 계산
    # CV = 표준편차(선택횟수) / 평균(선택횟수)
    # 완전히 균등하면 CV = 0.0 이 되며, 특정 전문가만 계속 선택하면 CV가 치솟아 0.3 한도를 초과합니다.
    counts_np = expert_counts.numpy()
    mean_count = counts_np.mean()
    std_count = counts_np.std()
    cv = (std_count / mean_count).item() if mean_count > 0 else 0.0
    
    print("\nMoE Expert Routing Analysis:")
    for i, pct in enumerate(expert_percentages):
        print(f"  Expert {i}: {pct * 100:.2f}% (count: {int(counts_np[i])})")
    print(f"  Load Balancing CV: {cv:.4f}")
    
    # 특정 전문가의 가용 빈도가 5%에 미치지 못하는 붕괴(Collapse) 사례를 판정합니다.
    collapsed_experts = []
    for i, pct in enumerate(expert_percentages):
        if pct < 0.05:
            collapsed_experts.append(i)
            print(f"  ⚠️ Expert {i} collapsed! (Usage: {pct * 100:.2f}%)")
            
    if not collapsed_experts:
        print("  ✅ All experts active. No expert collapse detected.")
        
    # -------------------------------------------------------------------------
    # [8단계] 자동화 평가 결과 JSON 리포트 디스크 영구 저장
    # -------------------------------------------------------------------------
    # PPL 50 미만, CV 0.3 미만, 그리고专家 낙오가 없어야 PASS로 판정합니다 (모에 평가 명세서 기준).
    report = {
        "step": step,
        "validation_cross_entropy": mean_loss,
        "validation_perplexity": val_ppl,
        "load_balancing_cv": cv,
        "expert_usages": expert_percentages,
        "collapsed_experts": collapsed_experts,
        "status": "pass" if (val_ppl < 50.0 and cv < 0.3 and len(collapsed_experts) == 0) else "fail"
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
    parser.add_argument("--checkpoint_pattern", type=str, default="moe_")
    parser.add_argument("--data_dir", type=str, default="train/data")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="drive_mock/reports/evaluation_report.json")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    evaluate(args)

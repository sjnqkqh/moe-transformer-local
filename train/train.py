import os
import sys
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

# 프로젝트 루트 경로를 참조하여 모델 모듈을 임포트하기 위해 sys.path에 추가합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.moe_transformer import MoETransformer
from model.moe_layer import MoETransformerBlock
from train.utils import (
    save_checkpoint,
    load_latest_checkpoint,
    log_metrics,
    log_event,
    init_experiment,
    complete_experiment
)

class NumpyDataset(Dataset):
    def __init__(self, npy_path: str):
        # 대용량 바이너리 파일을 통째로 메모리에 로드하지 않고 mmap_mode="r" (메모리 맵) 모드로 접근합니다.
        # 필요할 때 필요한 만큼만 하드디스크에서 읽어오므로 메모리 낭비를 줄입니다.
        self.data = np.load(npy_path, mmap_mode="r")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 입력(input_ids)과 타겟 레이블(labels)을 같은 토큰 시퀀스로 만듭니다.
        # 디코더 전용 트랜스포머 모델의 forward 함수 내부에서 자동으로 한 칸씩 밀어서(Shift) Loss를 계산해 줍니다.
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

class RoutingProfiler:
    """
    학습 과정 중에 전문가들의 사용 빈도를 모니터링하기 위한 라우터 프로파일러 훅(Hook) 객체.
    """
    def __init__(self):
        self.selections = []
        
    def hook_fn(self, module, input, output):
        # MoERouter.forward가 반환하는 세 가지 출력 중 1번째 값인 top_k_indices를 detach하여 캡처합니다.
        # output[1] shape: (S, k) - 각 토큰이 선택한 전문가 인덱스 정보
        self.selections.append(output[1].detach().cpu())
        
    def clear(self):
        # 로그를 한번 남긴 후에는 카운팅을 비워줍니다.
        self.selections = []
        
    def get_metrics(self):
        # 캡처된 기록이 없다면 기본 균등 수치 반환
        if not self.selections:
            return [0.25, 0.25, 0.25, 0.25], 0.0
            
        # 모든 배치/토큰의 전문가 선택 내역을 세로로 병합: (Total_Tokens, k)
        all_indices = torch.cat(self.selections, dim=0)
        total_tokens = all_indices.numel()
        
        # 각 전문가(0번 ~ 3번)가 선택된 횟수를 카운팅합니다.
        counts = torch.zeros(4)
        for idx in all_indices.view(-1):
            if idx.item() < 4:
                counts[idx.item()] += 1
                
        # 비율(%)로 변환
        usage = (counts / total_tokens).tolist()
        
        # 각 전문가 선택 빈도 분포의 Shannon Entropy를 구합니다.
        # 엔트로피가 최대값(약 1.386)에 가까울수록 전문가가 치우침 없이 완벽하게 균등 배분되고 있음을 뜻합니다.
        probs = counts / (counts.sum() + 1e-10)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        return usage, entropy

def train(args):
    # Accelerate 초기화: mixed_precision="bf16"은 A100 GPU에서 학습 속도를 몇 배 향상시키고 메모리를 아낍니다.
    # 단, 로컬 디버깅/M2 CPU 모드에서는 BF16 미지원으로 인해 mixed_precision="no" (FP32)로 실행합니다.
    mixed_precision = "no" if args.smoke_test else "bf16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    device = accelerator.device
    print(f"Device initialized: {device} (Mixed Precision: {mixed_precision})")
    
    # 체크포인트 및 로그를 보관할 저장 경로 설정
    ckpt_dir = os.path.join(args.project_dir, "checkpoints")
    log_dir = os.path.join(args.project_dir, "logs")
    
    # 1. 모델 객체 생성
    print("Instantiating MoE Transformer...")
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
    
    # 파라미터 상세 구성 정보 출력 (사용자 학습 안내 목적)
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        emb_params = model.token_embeddings.weight.numel()
        attn_params = sum(p.numel() for name, p in model.named_parameters() if "attention" in name)
        
        dense_ffn_params = 0
        moe_ffn_params = 0
        for name, p in model.named_parameters():
            if "ffn" in name and "attention" not in name:
                parts = name.split(".")
                if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    if layer_idx % 2 == 0:
                        moe_ffn_params += p.numel()
                    else:
                        dense_ffn_params += p.numel()
                        
        lm_head_params = model.lm_head.weight.numel()
        
        print("-" * 50)
        print("Model Architecture Parameter Breakdown:")
        print(f"  - Total Parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  - Token Embedding:       {emb_params:,} ({emb_params/1e6:.2f}M)")
        print(f"  - Attention (8 layers):  {attn_params:,} ({attn_params/1e6:.2f}M)")
        print(f"  - Dense FFN (4 layers):  {dense_ffn_params:,} ({dense_ffn_params/1e6:.2f}M)")
        print(f"  - MoE FFN (4 layers):    {moe_ffn_params:,} ({moe_ffn_params/1e6:.2f}M)")
        print(f"  - LM Head (untied):      {lm_head_params:,} ({lm_head_params/1e6:.2f}M)")
        print("-" * 50)
        
    # 2. 데이터 가공 파일 로드 및 배치 데이터로더 설정
    train_npy = os.path.join(args.data_dir, "train.npy")
    if not os.path.exists(train_npy):
        raise FileNotFoundError(f"Training dataset not found at {train_npy}. Please run prepare_data.py first.")
        
    dataset = NumpyDataset(train_npy)
    batch_size = 2 if args.smoke_test else args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 3. 최적화기(Optimizer) 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 학습률 스케줄러: 초기에 서서히 올렸다가 코사인 감쇄 곡선으로 학습률을 차츰 낮춥니다.
    max_steps = 10 if args.smoke_test else args.max_steps
    warmup_steps = 2 if args.smoke_test else args.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 1단계: 선형 웜업 구간 (Linear Warmup)
            return float(current_step) / float(max(1, warmup_steps))
        # 2단계: 코사인 감쇄 구간 (Cosine Decay)
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 4. 전문가 라우팅 모니터링을 위한 Forward 훅(Profiler) 연결
    profiler = RoutingProfiler()
    hooks = []
    for layer in model.layers:
        if isinstance(layer, MoETransformerBlock):
            # MoEFFN 모듈 내부의 MoERouter에 훅을 달아 토큰 분배값을 낚아챕니다.
            hooks.append(layer.ffn.router.register_forward_hook(profiler.hook_fn))
            
    # 5. Accelerate를 이용한 다중 장치(DDP/GPU/CPU) 및 연산 포장 준비
    # DDP 및 믹스드 프리시전 분산 처리에 맞게 모델, 데이터로더 등을 재배치합니다.
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # 6. 중간 중단 시점의 최근 체크포인트 자동 복원 검사
    pattern = f"moe_{args.run_id}"
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler, pattern)
    
    # 실험 정보 최초 등록 (중복 방지를 위해 step이 0이고 메인 머신인 경우에만 1회 기록)
    if start_step == 0 and accelerator.is_main_process:
        config = vars(args)
        config["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        init_experiment(log_dir, args.run_id, args.name, config)
        
    print(f"Starting training loop from step {start_step} to {max_steps}...")
    
    step = start_step
    prev_loss = None
    step_time = time.time()
    total_tokens_processed = 0
    
    model.train()
    
    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break
                
            input_ids, labels = batch
            
            optimizer.zero_grad()
            
            # [역전파 1] 순전파 (Forward Pass)
            logits, loss, main_loss, aux_loss, z_loss = model(input_ids, labels)
            
            # [역전파 2] 역전파 (Backward Pass)
            # DDP 분산 처리 및 BF16 스케일링을 자동으로 조율하며 그래디언트를 산출합니다.
            accelerator.backward(loss)
            
            # [역전파 3] 그래디언트 클리핑 (Gradient Clipping)
            # 가중치의 미분 크기 절대값을 1.0 한도로 잘라내어 경사도 폭발(Spike)을 미연에 방지합니다.
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
                
            # [역전파 4] 가중치 갱신 및 학습 스케줄러 스텝
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            # 속도 연산용 토큰 처리 수 누계
            num_tokens = input_ids.numel()
            total_tokens_processed += num_tokens
            
            # 메인 콘솔 경고 검사 (메인 머신 단독)
            if accelerator.is_main_process:
                # 이상 스파이크 점검: 직전 스텝보다 손실(loss)이 20% 이상 급격히 튀었는지 모니터링합니다.
                curr_loss_val = loss.item()
                if prev_loss is not None and curr_loss_val > prev_loss * 1.2:
                    log_event(log_dir, args.run_id, "loss_spike", {
                        "step": step,
                        "previous_loss": prev_loss,
                        "current_loss": curr_loss_val,
                        "grad_norm": grad_norm
                    })
                prev_loss = curr_loss_val
                
            # 주기적인 훈련 진행 로그 출력 및 파일 쓰기
            log_interval = 1 if args.smoke_test else args.log_every
            if step % log_interval == 0:
                # 분산 프로세스들 동기화 대기
                accelerator.wait_for_everyone()
                
                # 라우터 사용 비중 분석 후 훅 캐시 삭제
                expert_usage, router_entropy = profiler.get_metrics()
                profiler.clear()
                
                # 학습 처리 속도(tokens/second) 계산
                elapsed = time.time() - step_time
                tokens_per_sec = total_tokens_processed / max(1e-5, elapsed)
                
                # 속도계 갱신 초기화
                step_time = time.time()
                total_tokens_processed = 0
                
                gpu_memory_gb = 0.0
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                    
                if accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    main_l_val = main_loss.item()
                    aux_l_val = aux_loss.item()
                    z_l_val = z_loss.item()
                    total_l_val = loss.item()
                    ppl = np.exp(min(20, main_l_val)) # 오버플로우 방지 캡 적용
                    
                    metrics = {
                        "main_loss": main_l_val,
                        "aux_loss": aux_l_val,
                        "z_loss": z_l_val,
                        "total_loss": total_l_val,
                        "ppl": ppl,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "expert_usage": expert_usage,
                        "router_entropy": router_entropy,
                        "gpu_memory_gb": gpu_memory_gb,
                        "tokens_per_sec": tokens_per_sec,
                        "epoch_progress": step / max_steps
                    }
                    
                    # metrics.jsonl 파일에 한 줄의 JSON 텍스트 추가
                    log_metrics(log_dir, args.run_id, step, metrics)
                    print(f"Step {step}/{max_steps} | Loss: {total_l_val:.4f} | PPL: {ppl:.2f} | lr: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
                    
                    # 전문가 붕괴 점검: 가용 비중이 5% 미만인 전문가 발생 시 비상 경고 로그 기록
                    for exp_idx, usage_ratio in enumerate(expert_usage):
                        if usage_ratio < 0.05:
                            log_event(log_dir, args.run_id, "expert_collapse", {
                                "step": step,
                                "expert_id": exp_idx,
                                "usage_ratio": usage_ratio,
                                "all_usages": expert_usage
                            })
                            
            # 모델 체크포인트 보관 (중간 저장 및 학습 완료 최종 저장)
            save_interval = 2 if args.smoke_test else args.save_every
            if step % save_interval == 0 or step == max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, loss.item(), pattern)
                    log_event(log_dir, args.run_id, "checkpoint", {
                        "step": step,
                        "loss": loss.item()
                    })
                    
    # 훅 제거를 통해 메모리 누수를 완전히 예방합니다.
    for h in hooks:
        h.remove()
        
    # 실험 종결 등록
    if accelerator.is_main_process:
        print("Training complete!")
        final_metrics = {
            "final_step": step,
            "final_loss": loss.item() if 'loss' in locals() else -1.0
        }
        complete_experiment(log_dir, args.run_id, final_metrics)
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Unique run ID, e.g. r001")
    parser.add_argument("--name", type=str, default="moe_baseline", help="Description name")
    parser.add_argument("--data_dir", type=str, default="train/data")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/output")
    parser.add_argument("--project_dir", type=str, default="drive_mock")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    train(args)

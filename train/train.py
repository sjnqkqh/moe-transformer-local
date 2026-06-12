import os
import sys
import time
import argparse
import datetime
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import wandb

# 프로젝트 루트 경로를 참조하여 모델 모듈을 임포트하기 위해 sys.path에 추가합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig
from train.utils import (
    save_checkpoint,
    load_latest_checkpoint,
    log_metrics,
    log_event,
    init_experiment,
    complete_experiment,
    NumpyDataset,
)


class EarlyStopping:
    """조기 중단 (Early Stopping) — 검증 손실이 개선되지 않으면 학습 중단"""

    def __init__(self, patience=5, delta=1e-3, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.best_model_state = None
        self.best_step = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, step):
        score = -val_loss  # loss는 낮을수록 좋음

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_step = step
            if self.verbose:
                print(
                    f"  [EarlyStopping] Step {step}: 최초 저장 (val_loss={val_loss:.4f})"
                )
            return False

        if val_loss > self.best_loss - self.delta:
            # 개선 없음
            self.counter += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] Step {step}: 개선 없음 {self.counter}/{self.patience} "
                    f"(val_loss={val_loss:.4f}, best={self.best_loss:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_state)
                if self.verbose:
                    print(
                        f"  ★ Early Stopping 발동! Step {self.best_step}의 가중치로 복원 "
                        f"(val_loss={self.best_loss:.4f})"
                    )
                return True
        else:
            # 개선됨
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_step = step
            self.counter = 0
            if self.verbose:
                print(
                    f"  [EarlyStopping] Step {step}: 개선! 저장 (val_loss={val_loss:.4f})"
                )

        return False


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

    if args.wandb and accelerator.is_main_process:
        wandb.init(project="korean-dense-chatbot", name=args.run_id, config=vars(args))

    # 1. 모델 객체 생성
    print("Instantiating Dense Transformer...")
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)
    print(
        f"Loaded tokenizer from {args.tokenizer_dir}. Dynamically set vocab_size to {vocab_size}"
    )

    config = DenseTransformerConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=3072,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    )
    model = DenseTransformer(config)

    # 파라미터 상세 구성 정보 출력
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        emb_params = model.token_embeddings.weight.numel()
        attn_params = sum(
            p.numel() for name, p in model.named_parameters() if "attention" in name
        )
        ffn_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "ffn" in name and "attention" not in name
        )
        lm_head_params = model.lm_head.weight.numel()

        print("-" * 50)
        print("Model Architecture Parameter Breakdown:")
        print(
            f"  - Total Parameters:      {total_params:,} ({total_params / 1e6:.2f}M)"
        )
        print(f"  - Token Embedding:       {emb_params:,} ({emb_params / 1e6:.2f}M)")
        print(f"  - Attention (12 layers): {attn_params:,} ({attn_params / 1e6:.2f}M)")
        print(f"  - FFN (12 layers):       {ffn_params:,} ({ffn_params / 1e6:.2f}M)")
        print(
            f"  - LM Head (untied):      {lm_head_params:,} ({lm_head_params / 1e6:.2f}M)"
        )
        print("-" * 50)

    # 2. 학습 데이터 로드
    train_npy = os.path.join(args.data_dir, "train.npy")
    if not os.path.exists(train_npy):
        raise FileNotFoundError(
            f"Training dataset not found at {train_npy}. Please run prepare_data.py first."
        )

    train_dataset = NumpyDataset(train_npy)
    batch_size = 2 if args.smoke_test else args.batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # 3. 검증 데이터 로드 (val.npy가 있으면 사용, 없으면 건너뜀)
    val_npy = os.path.join(args.data_dir, "val.npy")
    val_loader = None
    if os.path.exists(val_npy):
        val_dataset = NumpyDataset(val_npy)
        val_batch_size = batch_size * 2  # 검증은 메모리가 여유로우므로 2배
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False
        )
        print(f"Loaded validation set: {len(val_dataset):,} blocks from {val_npy}")
    else:
        print(
            f"⚠️ Validation set not found at {val_npy}. Early Stopping을 사용하려면 prepare_data.py로 val.npy를 생성하세요."
        )

    # 4. 최적화기(Optimizer) 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 만약 --epochs가 지정되었다면 max_steps를 자동 계산합니다.
    if args.epochs is not None:
        args.max_steps = len(train_loader) * args.epochs
        if accelerator.is_main_process:
            print(
                f"Calculated max_steps: {args.max_steps} for {args.epochs} epochs (1 epoch = {len(train_loader)} steps)"
            )

    # 학습률 스케줄러
    max_steps = 10 if args.smoke_test else args.max_steps
    warmup_steps = 2 if args.smoke_test else args.warmup_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Accelerate 준비
    if val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    # 6. 체크포인트 복원
    pattern = f"dense_{args.run_id}"
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler, pattern)

    # 실험 정보 등록
    if start_step == 0 and accelerator.is_main_process:
        config = vars(args)
        config["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        init_experiment(log_dir, args.run_id, args.name, config)

    print(f"Starting training loop from step {start_step} to {max_steps}...")

    # 7. Early Stopping 초기화
    early_stopping = EarlyStopping(
        patience=args.patience,
        delta=args.min_delta,
        verbose=accelerator.is_main_process,
    )

    step = start_step
    prev_loss = None
    step_time = time.time()
    total_tokens_processed = 0
    best_val_loss = float("inf")
    best_checkpoint_saved = False

    model.train()

    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= max_steps:
                break

            input_ids, labels = batch

            optimizer.zero_grad()

            logits, loss, main_loss = model(input_ids, labels)
            accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()

            optimizer.step()
            scheduler.step()

            step += 1

            # 속도 계산
            num_tokens = input_ids.numel()
            total_tokens_processed += num_tokens

            # Loss spike 감지
            if accelerator.is_main_process:
                curr_loss_val = loss.item()
                if prev_loss is not None and curr_loss_val > prev_loss * 1.2:
                    log_event(
                        log_dir,
                        args.run_id,
                        "loss_spike",
                        {
                            "step": step,
                            "previous_loss": prev_loss,
                            "current_loss": curr_loss_val,
                            "grad_norm": grad_norm,
                        },
                    )
                prev_loss = curr_loss_val

            # 학습 로그 출력
            log_interval = 1 if args.smoke_test else args.log_every
            if step % log_interval == 0:
                accelerator.wait_for_everyone()

                elapsed = time.time() - step_time
                tokens_per_sec = total_tokens_processed / max(1e-5, elapsed)

                step_time = time.time()
                total_tokens_processed = 0

                gpu_memory_gb = 0.0
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9

                if accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    main_l_val = main_loss.item()
                    total_l_val = loss.item()
                    ppl = np.exp(min(20, main_l_val))

                    metrics = {
                        "main_loss": main_l_val,
                        "total_loss": total_l_val,
                        "ppl": ppl,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "gpu_memory_gb": gpu_memory_gb,
                        "tokens_per_sec": tokens_per_sec,
                        "epoch_progress": step / max_steps,
                    }

                    log_metrics(log_dir, args.run_id, step, metrics)
                    print(
                        f"Step {step}/{max_steps} | Loss: {total_l_val:.4f} | PPL: {ppl:.2f} | lr: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s"
                    )

                    if args.wandb:
                        wandb.log(
                            {
                                "train/loss": total_l_val,
                                "train/main_loss": main_l_val,
                                "train/ppl": ppl,
                                "train/lr": lr,
                                "train/grad_norm": grad_norm,
                                "system/gpu_memory_gb": gpu_memory_gb,
                                "system/tokens_per_sec": tokens_per_sec,
                                "system/epoch_progress": step / max_steps,
                            },
                            step=step,
                        )

            # 체크포인트 저장 (주기적)
            save_interval = 2 if args.smoke_test else args.save_every
            if step % save_interval == 0 or step == max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(
                        ckpt_dir,
                        model,
                        optimizer,
                        scheduler,
                        step,
                        loss.item(),
                        pattern,
                    )
                    log_event(
                        log_dir,
                        args.run_id,
                        "checkpoint",
                        {"step": step, "loss": loss.item()},
                    )

            # --- 검증(Validation) ---
            if (
                val_loader is not None
                and args.val_every > 0
                and step % args.val_every == 0
            ):
                model.eval()
                total_val_loss = 0.0
                num_val_batches = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids, val_labels = val_batch
                        _, val_loss, val_main_loss = model(val_input_ids, val_labels)
                        total_val_loss += val_loss.item()
                        num_val_batches += 1

                avg_val_loss = total_val_loss / max(1, num_val_batches)
                avg_val_ppl = np.exp(min(20, avg_val_loss))

                if accelerator.is_main_process:
                    print(
                        f"  ▶ Validation: step {step} | val_loss={avg_val_loss:.4f} | val_ppl={avg_val_ppl:.2f}"
                    )

                    # Best validation checkpoint 저장
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_checkpoint_path = os.path.join(
                            ckpt_dir, f"dense_{args.run_id}_best.pt"
                        )
                        accelerator.wait_for_everyone()
                        uw = accelerator.unwrap_model(model)
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": uw.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "val_loss": avg_val_loss,
                                "val_ppl": avg_val_ppl,
                            },
                            best_checkpoint_path,
                        )
                        best_checkpoint_saved = True
                        print(
                            f"  ⭐ Best validation checkpoint saved (val_loss={avg_val_loss:.4f})"
                        )

                    # Wandb에 validation metrics 로깅
                    if args.wandb:
                        wandb.log(
                            {
                                "val/loss": avg_val_loss,
                                "val/ppl": avg_val_ppl,
                            },
                            step=step,
                        )

                    # Log to file
                    log_metrics(
                        log_dir,
                        args.run_id,
                        step,
                        {
                            "val_loss": avg_val_loss,
                            "val_ppl": avg_val_ppl,
                        },
                    )

                # Early Stopping 체크 (validation 기준)
                if early_stopping(
                    avg_val_loss,
                    (
                        accelerator.unwrap_model(model)
                        if hasattr(model, "module")
                        else model
                    ),
                    step,
                ):
                    # Early Stopping 발동 — best checkpoint가 있으면 그걸로 저장
                    if best_checkpoint_saved:
                        final_path = os.path.join(
                            ckpt_dir, f"dense_{args.run_id}_final_early_stop.pt"
                        )
                        shutil_copy_if_exists(best_checkpoint_path, final_path)
                    accelerator.wait_for_everyone()
                    break  # 학습 종료

                model.train()

    # 훈련 완료 처리
    if accelerator.is_main_process:
        print("Training complete!")

        # Early Stopping으로 종료되지 않은 경우에도 best weight로 복원
        if (
            val_loader is not None
            and not early_stopping.early_stop
            and best_checkpoint_saved
        ):
            print(
                f"Early Stopping 미발동. Best checkpoint (step {early_stopping.best_step}, val_loss={early_stopping.best_loss:.4f})로 복원합니다."
            )
            model.load_state_dict(early_stopping.best_model_state)

        final_metrics = {
            "final_step": step,
            "final_loss": loss.item() if "loss" in locals() else -1.0,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "early_stopped": early_stopping.early_stop,
            "best_step": early_stopping.best_step,
        }
        complete_experiment(log_dir, args.run_id, final_metrics)
        if args.wandb:
            wandb.finish()
        print("=" * 60)


def shutil_copy_if_exists(src, dst):
    """파일이 존재하면 복사, 없으면 무시"""
    import shutil

    if os.path.exists(src):
        shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, required=True, help="Unique run ID, e.g. r001"
    )
    parser.add_argument(
        "--name", type=str, default="dense_baseline", help="Description name"
    )
    parser.add_argument("--data_dir", type=str, default="train/data")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/output")
    parser.add_argument("--project_dir", type=str, default="drive_mock")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="학습할 에폭 수 (설정 시 max_steps 무시)",
    )
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.15, help="드롭아웃 비율")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--wandb", action="store_true", help="wandb 클라우드 로깅 활성화"
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=500)

    # --- Early Stopping 관련 인자 ---
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Validation loss 개선 없이 기다릴 step 횟수 (val_every 단위)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-3,
        help="개선으로 인정할 최소 val_loss 변화량",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=500,
        help="몇 step마다 validation을 실행할지 (0이면 validation 스킵)",
    )

    args = parser.parse_args()

    train(args)

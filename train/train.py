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

KST = datetime.timezone(datetime.timedelta(hours=9))


def make_kst_run_dir(base_dir: str, run_id: str) -> str:
    """한국 시간 기반 서브폴더를 project_dir 아래에 생성하고 경로를 반환합니다."""
    now_kst = datetime.datetime.now(KST)
    stamp = now_kst.strftime("%Y_%m_%d_%H_%M")
    run_dir = os.path.join(base_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class EarlyStopping:
    """조기 중단 — 검증 손실이 개선되지 않으면 학습 중단"""

    def __init__(self, patience=8, delta=1e-3, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.best_model_state = None
        self.best_step = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, step):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_step = step
            if self.verbose:
                print(f"  [EarlyStopping] Step {step}: 최초 저장 (val_loss={val_loss:.4f})")
            return False

        if val_loss > self.best_loss - self.delta:
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
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_step = step
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Step {step}: 개선! 저장 (val_loss={val_loss:.4f})")

        return False


def train(args):
    mixed_precision = "no" if args.smoke_test else "bf16"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )

    device = accelerator.device
    print(f"Device initialized: {device} (Mixed Precision: {mixed_precision})")

    # KST 기반 실행 디렉토리 결정 —————————————————————————————————————
    # 새 학습(체크포인트 없음)이면 project_dir/{KST 타임스탬프}/ 아래에 저장.
    # 재개 시엔 --project_dir 에 타임스탬프 폴더까지 포함한 경로를 직접 지정.
    effective_project_dir = args.project_dir
    pattern = f"dense_{args.run_id}"
    probe_ckpt_dir = os.path.join(args.project_dir, "checkpoints")
    probe_ckpts = []
    if os.path.isdir(probe_ckpt_dir):
        import glob as _glob
        probe_ckpts = _glob.glob(os.path.join(probe_ckpt_dir, f"{pattern}*.pt"))

    if not probe_ckpts and not args.smoke_test:
        # 신규 학습 → KST 서브폴더 생성
        if accelerator.is_main_process:
            effective_project_dir = make_kst_run_dir(args.project_dir, args.run_id)
            print(f"📁 새 학습 실행 디렉토리(KST): {effective_project_dir}")
        # 모든 프로세스가 같은 경로를 써야 하므로 broadcast
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            path_bytes = effective_project_dir.encode()
            path_tensor = torch.tensor(
                list(path_bytes) + [0] * (512 - len(path_bytes)), dtype=torch.uint8
            ).to(device)
            dist.broadcast(path_tensor, src=0)
            effective_project_dir = bytes(
                path_tensor.cpu().tolist()
            ).rstrip(b"\x00").decode()

    ckpt_dir = os.path.join(effective_project_dir, "checkpoints")
    log_dir = os.path.join(effective_project_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.wandb and accelerator.is_main_process:
        wandb.init(project="korean-dense-chatbot", name=args.run_id, config=vars(args))

    # 1. 모델 생성 ——————————————————————————————————————————————————
    print("Instantiating Dense Transformer...")
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer from {args.tokenizer_dir}. vocab_size={vocab_size}")

    # config.py 기본값(720M)을 그대로 사용 — 명시적으로 덮어쓸 값만 지정
    config = DenseTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    )
    model = DenseTransformer(config)

    # Gradient checkpointing — A100 40GB에서 720M+block2048 학습 필수
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("✅ Gradient checkpointing 활성화 (활성화 메모리 ~4배 절감)")

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        emb_params = model.token_embeddings.weight.numel()
        attn_params = sum(p.numel() for name, p in model.named_parameters() if "attention" in name)
        ffn_params = sum(p.numel() for name, p in model.named_parameters() if "ffn" in name and "attention" not in name)
        lm_head_params = model.lm_head.weight.numel()

        print("-" * 50)
        print("Model Architecture Parameter Breakdown:")
        print(f"  - Total Parameters:        {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  - d_model / n_layers:      {config.d_model} / {config.n_layers}")
        print(f"  - Token Embedding:         {emb_params:,} ({emb_params / 1e6:.2f}M)")
        print(f"  - Attention ({config.n_layers} layers): {attn_params:,} ({attn_params / 1e6:.2f}M)")
        print(f"  - FFN     ({config.n_layers} layers): {ffn_params:,} ({ffn_params / 1e6:.2f}M)")
        print(f"  - LM Head (untied):        {lm_head_params:,} ({lm_head_params / 1e6:.2f}M)")
        print("-" * 50)

    # 2. 학습 데이터 ——————————————————————————————————————————————
    train_npy = os.path.join(args.data_dir, "train.npy")
    if not os.path.exists(train_npy):
        raise FileNotFoundError(f"Training dataset not found at {train_npy}. Run prepare_data.py first.")

    train_dataset = NumpyDataset(train_npy)
    batch_size = 2 if args.smoke_test else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. 검증 데이터 ——————————————————————————————————————————————
    val_npy = os.path.join(args.data_dir, "val.npy")
    val_loader = None
    if os.path.exists(val_npy):
        val_dataset = NumpyDataset(val_npy)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
        )
        print(f"Loaded validation set: {len(val_dataset):,} blocks from {val_npy}")
    else:
        print(f"⚠️ val.npy not found at {val_npy}. Early Stopping 비활성화.")

    # 4. 옵티마이저 & 스케줄러 —————————————————————————————————————
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    if args.epochs is not None:
        # max_steps는 옵티마이저 step 단위 (grad_accum 반영)
        micro_steps_per_epoch = len(train_loader)
        opt_steps_per_epoch = max(1, micro_steps_per_epoch // args.grad_accum)
        args.max_steps = opt_steps_per_epoch * args.epochs
        if accelerator.is_main_process:
            print(
                f"Calculated max_steps: {args.max_steps} optimizer steps for {args.epochs} epoch(s) "
                f"(1 epoch = {opt_steps_per_epoch} opt steps = {micro_steps_per_epoch} micro-batches)"
            )

    max_steps = 10 if args.smoke_test else args.max_steps
    warmup_steps = 2 if args.smoke_test else args.warmup_steps
    min_lr_ratio = args.min_lr / max(args.lr, 1e-10)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_val = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Accelerate 준비 ——————————————————————————————————————————
    if val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    # 6. 체크포인트 복원 ——————————————————————————————————————————
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler, pattern)

    if start_step == 0 and accelerator.is_main_process:
        cfg = vars(args)
        cfg["timestamp"] = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
        cfg["effective_project_dir"] = effective_project_dir
        init_experiment(log_dir, args.run_id, args.name, cfg)

    print(f"Starting training from step {start_step} → {max_steps}...")
    print(f"  ckpt_dir : {ckpt_dir}")
    print(f"  log_dir  : {log_dir}")

    # 7. Early Stopping ———————————————————————————————————————————
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
    best_checkpoint_path = None
    best_checkpoint_saved = False

    model.train()

    grad_norm = 0.0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            input_ids, labels = batch

            # accelerate.accumulate: sync_gradients=True 인 마지막 micro-step에만
            # optimizer/scheduler가 실제로 step하고, clip_grad_norm_도 그때만 적용
            with accelerator.accumulate(model):
                logits, loss, main_loss = model(input_ids, labels)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip
                    )
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 토큰 카운트는 micro-step 단위 누적
            num_tokens = input_ids.numel()
            total_tokens_processed += num_tokens

            # 옵티마이저 step이 실제로 일어났을 때만 step 카운터 증가 + 로깅
            if not accelerator.sync_gradients:
                continue

            step += 1

            if accelerator.is_main_process:
                curr_loss_val = loss.item()
                if prev_loss is not None and curr_loss_val > prev_loss * 1.2:
                    log_event(
                        log_dir, args.run_id, "loss_spike",
                        {"step": step, "previous_loss": prev_loss, "current_loss": curr_loss_val, "grad_norm": grad_norm},
                    )
                prev_loss = curr_loss_val

            # 학습 로그
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
                        f"Step {step}/{max_steps} | Loss: {total_l_val:.4f} | PPL: {ppl:.2f} "
                        f"| lr: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s"
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
                            },
                            step=step,
                        )

            # 체크포인트 저장
            save_interval = 2 if args.smoke_test else args.save_every
            if step % save_interval == 0 or step == max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, loss.item(), pattern)
                    log_event(log_dir, args.run_id, "checkpoint", {"step": step, "loss": loss.item()})

            # Validation
            if val_loader is not None and args.val_every > 0 and step % args.val_every == 0:
                model.eval()
                total_val_loss = 0.0
                num_val_batches = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids, val_labels = val_batch
                        _, val_loss, _ = model(val_input_ids, val_labels)
                        total_val_loss += val_loss.item()
                        num_val_batches += 1

                avg_val_loss = total_val_loss / max(1, num_val_batches)
                avg_val_ppl = np.exp(min(20, avg_val_loss))

                if accelerator.is_main_process:
                    print(f"  ▶ Validation: step {step} | val_loss={avg_val_loss:.4f} | val_ppl={avg_val_ppl:.2f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_checkpoint_path = os.path.join(ckpt_dir, f"dense_{args.run_id}_best.pt")
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
                        print(f"  ⭐ Best validation checkpoint saved (val_loss={avg_val_loss:.4f})")

                    if args.wandb:
                        wandb.log({"val/loss": avg_val_loss, "val/ppl": avg_val_ppl}, step=step)

                    log_metrics(log_dir, args.run_id, step, {"val_loss": avg_val_loss, "val_ppl": avg_val_ppl})

                should_stop = early_stopping(
                    avg_val_loss,
                    accelerator.unwrap_model(model) if hasattr(model, "module") else model,
                    step,
                )
                if should_stop:
                    if best_checkpoint_saved and best_checkpoint_path:
                        import shutil
                        final_path = os.path.join(ckpt_dir, f"dense_{args.run_id}_final_early_stop.pt")
                        shutil.copy2(best_checkpoint_path, final_path)
                    accelerator.wait_for_everyone()
                    break

                model.train()

    if accelerator.is_main_process:
        print("Training complete!")

        if val_loader is not None and not early_stopping.early_stop and best_checkpoint_saved:
            print(
                f"Early Stopping 미발동. Best checkpoint "
                f"(step {early_stopping.best_step}, val_loss={early_stopping.best_loss:.4f})로 복원합니다."
            )
            model.load_state_dict(early_stopping.best_model_state)

        final_metrics = {
            "final_step": step,
            "final_loss": loss.item() if "loss" in locals() else -1.0,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "early_stopped": early_stopping.early_stop,
            "best_step": early_stopping.best_step,
            "effective_project_dir": effective_project_dir,
        }
        complete_experiment(log_dir, args.run_id, final_metrics)
        if args.wandb:
            wandb.finish()
        print(f"모든 파일 저장 위치: {effective_project_dir}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Unique run ID (e.g. v8_720m)")
    parser.add_argument("--name", type=str, default="dense_baseline", help="Description name")
    parser.add_argument("--data_dir", type=str, default="train/data")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/output")
    parser.add_argument("--project_dir", type=str, default="drive_mock",
                        help="저장 루트. 신규 학습 시 KST 타임스탬프 서브폴더 자동 생성. "
                             "재개 시 타임스탬프 폴더까지 포함한 전체 경로 지정.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="per-device micro-batch (실효 배치 = batch_size × grad_accum)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="gradient accumulation steps (메모리 절약용 micro-batch 반복)")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="gradient checkpointing 활성화 (활성화 메모리 ~4배 절감, 속도 -25%%)")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Cosine decay 최소 LR")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None, help="에폭 수 (설정 시 max_steps 무시)")
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--dropout", type=float, default=0.0, help="사전학습=0.0, SFT=0.05")
    parser.add_argument("--val_every", type=int, default=1000,
                        help="몇 step마다 validation 실행 (0이면 스킵)")
    parser.add_argument("--patience", type=int, default=8,
                        help="val_loss 개선 없이 기다릴 횟수 (val_every 단위)")
    parser.add_argument("--min_delta", type=float, default=1e-3,
                        help="개선으로 인정할 최소 val_loss 변화량")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="W&B 클라우드 로깅 활성화")

    args = parser.parse_args()
    train(args)

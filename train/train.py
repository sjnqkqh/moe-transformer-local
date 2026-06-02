import os
import sys
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

# Add current workspace directory to sys.path
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
        self.data = np.load(npy_path, mmap_mode="r")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # inputs and labels are identical; model's internal shift will align them
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

class RoutingProfiler:
    """
    Hook to capture expert routing decisions and logit values in real time.
    """
    def __init__(self):
        self.selections = []
        
    def hook_fn(self, module, input, output):
        # output is (top_k_probs, top_k_indices, router_logits)
        # We capture the chosen expert indices: (S, k)
        self.selections.append(output[1].detach().cpu())
        
    def clear(self):
        self.selections = []
        
    def get_metrics(self):
        if not self.selections:
            return [0.25, 0.25, 0.25, 0.25], 0.0
            
        all_indices = torch.cat(self.selections, dim=0) # (S_all, k)
        total_tokens = all_indices.numel()
        
        counts = torch.zeros(4)
        for idx in all_indices.view(-1):
            if idx.item() < 4:
                counts[idx.item()] += 1
                
        # Usage ratio
        usage = (counts / total_tokens).tolist()
        
        # Shannon Entropy
        probs = counts / (counts.sum() + 1e-10)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        return usage, entropy

def train(args):
    # Initialize Accelerator
    # If in smoke_test, turn off mixed precision to run reliably on CPU/MPS
    mixed_precision = "no" if args.smoke_test else "bf16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    device = accelerator.device
    print(f"Device initialized: {device} (Mixed Precision: {mixed_precision})")
    
    # Paths config
    ckpt_dir = os.path.join(args.project_dir, "checkpoints")
    log_dir = os.path.join(args.project_dir, "logs")
    
    # 1. Instantiate Model
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
    
    # Calculate parameter counts for user inspection (Only on main process)
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        emb_params = model.token_embeddings.weight.numel()
        attn_params = sum(p.numel() for name, p in model.named_parameters() if "attention" in name)
        
        # Group FFN parameters by layer indices (even layers = MoE, odd layers = Dense)
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
        
    # 2. Setup Dataset & DataLoader
    train_npy = os.path.join(args.data_dir, "train.npy")
    if not os.path.exists(train_npy):
        raise FileNotFoundError(f"Training dataset not found at {train_npy}. Please run prepare_data.py first.")
        
    dataset = NumpyDataset(train_npy)
    # Adjust batch size in smoke test
    batch_size = 2 if args.smoke_test else args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 3. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Simple linear warmup with cosine decay
    max_steps = 10 if args.smoke_test else args.max_steps
    warmup_steps = 2 if args.smoke_test else args.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 4. Attach Routing Profiler Hooks
    profiler = RoutingProfiler()
    hooks = []
    for layer in model.layers:
        if isinstance(layer, MoETransformerBlock):
            hooks.append(layer.ffn.router.register_forward_hook(profiler.hook_fn))
            
    # 5. Prepare under Accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # 6. Resume from Checkpoint if exists
    # Pattern: moe_run_id (e.g. moe_r001)
    pattern = f"moe_{args.run_id}"
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler, pattern)
    
    # Register/Initialize Experiment
    if start_step == 0 and accelerator.is_main_process:
        config = vars(args)
        config["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        init_experiment(log_dir, args.run_id, args.name, config)
        
    print(f"Starting training loop from step {start_step} to {max_steps}...")
    
    # Tracking variables
    step = start_step
    prev_loss = None
    step_time = time.time()
    total_tokens_processed = 0
    
    model.train()
    
    # Loop over epochs
    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break
                
            input_ids, labels = batch # Shape: (B, T)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, loss, main_loss, aux_loss, z_loss = model(input_ids, labels)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Clip gradients
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
                
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            # Count tokens
            num_tokens = input_ids.numel()
            total_tokens_processed += num_tokens
            
            # Anomalies monitoring (Only on main process)
            if accelerator.is_main_process:
                # 1. Loss Spike check (increase of > 20% between consecutive steps)
                curr_loss_val = loss.item()
                if prev_loss is not None and curr_loss_val > prev_loss * 1.2:
                    log_event(log_dir, args.run_id, "loss_spike", {
                        "step": step,
                        "previous_loss": prev_loss,
                        "current_loss": curr_loss_val,
                        "grad_norm": grad_norm
                    })
                prev_loss = curr_loss_val
                
            # Log Metrics (every log_every steps or in smoke test every step)
            log_interval = 1 if args.smoke_test else args.log_every
            if step % log_interval == 0:
                # Synchronize to get accurate time and routing info
                accelerator.wait_for_everyone()
                
                # Fetch routing metrics
                expert_usage, router_entropy = profiler.get_metrics()
                profiler.clear()
                
                # Calculate speed
                elapsed = time.time() - step_time
                tokens_per_sec = total_tokens_processed / max(1e-5, elapsed)
                
                # Reset counters for speed calculation
                step_time = time.time()
                total_tokens_processed = 0
                
                # Get max memory used
                gpu_memory_gb = 0.0
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                    
                if accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    main_l_val = main_loss.item()
                    aux_l_val = aux_loss.item()
                    z_l_val = z_loss.item()
                    total_l_val = loss.item()
                    ppl = np.exp(min(20, main_l_val)) # cap ppl calculation to avoid overflow
                    
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
                    
                    log_metrics(log_dir, args.run_id, step, metrics)
                    print(f"Step {step}/{max_steps} | Loss: {total_l_val:.4f} | PPL: {ppl:.2f} | lr: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
                    
                    # 2. Expert Collapse Check (any expert used < 5%)
                    for exp_idx, usage_ratio in enumerate(expert_usage):
                        if usage_ratio < 0.05:
                            log_event(log_dir, args.run_id, "expert_collapse", {
                                "step": step,
                                "expert_id": exp_idx,
                                "usage_ratio": usage_ratio,
                                "all_usages": expert_usage
                            })
                            
            # Save Checkpoint (every save_every steps or at end)
            save_interval = 2 if args.smoke_test else args.save_every
            if step % save_interval == 0 or step == max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, loss.item(), pattern)
                    log_event(log_dir, args.run_id, "checkpoint", {
                        "step": step,
                        "loss": loss.item()
                    })
                    
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    # Complete Run
    if accelerator.is_main_process:
        print("Training complete!")
        # Record final stats
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

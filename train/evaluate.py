import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add current workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.moe_transformer import MoETransformer
from model.moe_layer import MoETransformerBlock

class NumpyDataset(Dataset):
    def __init__(self, npy_path: str):
        self.data = np.load(npy_path, mmap_mode="r")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

def evaluate(args):
    print("=" * 60)
    print("📊 MoE Transformer — Model Evaluation")
    print("=" * 60)
    
    # 1. Instantiate Model
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
    
    # 2. Load Checkpoint
    ckpts = glob.glob(os.path.join(args.ckpt_dir, f"{args.checkpoint_pattern}*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files matching '{args.checkpoint_pattern}' found in {args.ckpt_dir}")
        
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_ckpt}...")
    # Add weights_only=False for compatibility with PyTorch 2.6+ training states
    checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    
    # Clean up state dict keys if wrapped under DDP
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
    
    # Pick local device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running evaluation on: {device}")
    model.to(device)
    model.eval()
    
    # 3. Setup Dataset
    val_npy = os.path.join(args.data_dir, "val.npy")
    if not os.path.exists(val_npy):
        raise FileNotFoundError(f"Validation dataset not found at {val_npy}. Please run prepare_data.py first.")
        
    dataset = NumpyDataset(val_npy)
    batch_size = 2 if args.smoke_test else args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Hook Routers to capture expert assignments
    selected_indices = []
    def hook_fn(module, input, output):
        selected_indices.append(output[1].detach().cpu())
        
    hooks = []
    for layer in model.layers:
        if isinstance(layer, MoETransformerBlock):
            hooks.append(layer.ffn.router.register_forward_hook(hook_fn))
            
    # 5. Evaluation loop
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # Forward pass
            logits, loss, main_loss, aux_loss, z_loss = model(input_ids, labels)
            
            # Count actual tokens evaluated (excluding sequence end since shifted logits are size T-1)
            num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
            total_loss += main_loss.item() * num_tokens
            total_tokens += num_tokens
            total_batches += 1
            
            # Terminate early in smoke test
            if args.smoke_test and total_batches >= 5:
                break
                
    # Unregister hooks
    for h in hooks:
        h.remove()
        
    # 6. Compute metrics
    mean_loss = total_loss / max(1, total_tokens)
    val_ppl = np.exp(mean_loss)
    print(f"  Validation Loss: {mean_loss:.4f}")
    print(f"  Validation PPL:  {val_ppl:.4f}")
    
    # Analyze expert distributions
    all_indices = torch.cat(selected_indices, dim=0) # (S_all, k)
    expert_counts = torch.zeros(4)
    for idx in all_indices.view(-1):
        if idx.item() < 4:
            expert_counts[idx.item()] += 1
            
    total_selections = expert_counts.sum().item()
    expert_percentages = (expert_counts / total_selections).tolist()
    
    counts_np = expert_counts.numpy()
    mean_count = counts_np.mean()
    std_count = counts_np.std()
    
    # CV = std / mean
    cv = (std_count / mean_count).item() if mean_count > 0 else 0.0
    
    print("\nMoE Expert Routing Analysis:")
    for i, pct in enumerate(expert_percentages):
        print(f"  Expert {i}: {pct * 100:.2f}% (count: {int(counts_np[i])})")
    print(f"  Load Balancing CV: {cv:.4f}")
    
    collapsed_experts = []
    for i, pct in enumerate(expert_percentages):
        if pct < 0.05:
            collapsed_experts.append(i)
            print(f"  ⚠️ Expert {i} collapsed! (Usage: {pct * 100:.2f}%)")
            
    if not collapsed_experts:
        print("  ✅ All experts active. No expert collapse detected.")
        
    # Write Evaluation Report
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

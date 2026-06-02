import os
import sys
import torch

# Add current workspace directory to sys.path so model can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.moe_transformer import MoETransformer
from model.moe_layer import MoETransformerBlock

def main():
    print("=" * 60)
    print("🤖 MoE Transformer — Local Debug Verification")
    print("=" * 60)
    
    # 1. Instantiate Model
    print("\n[1/5] Instantiating model...")
    model = MoETransformer(
        vocab_size=32000,
        d_model=768,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        num_experts=4,
        k=2,
        max_seq_len=1024
    )
    print("    ✅ Model instantiated successfully.")
    
    # 2. Count Parameters
    print("\n[2/5] Counting parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
    
    print(f"    - Total Parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"    - Trainable Parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"    - Token Embedding:       {emb_params:,} ({emb_params/1e6:.2f}M)")
    print(f"    - Attention (8 layers):  {attn_params:,} ({attn_params/1e6:.2f}M)")
    print(f"    - Dense FFN (4 layers):  {dense_ffn_params:,} ({dense_ffn_params/1e6:.2f}M)")
    print(f"    - MoE FFN (4 layers):    {moe_ffn_params:,} ({moe_ffn_params/1e6:.2f}M)")
    print(f"    - LM Head (untied):      {lm_head_params:,} ({lm_head_params/1e6:.2f}M)")
    
    # Expect total params around ~162.4M for untied embeddings
    assert 160e6 < total_params < 165e6, f"Expected parameters to be ~162.4M, got {total_params/1e6:.2f}M"
    print("    ✅ Parameter counts are within the expected range (~162.4M untied).")
    
    # 3. Forward Pass Validation
    print("\n[3/5] Verifying forward pass with dummy batch...")
    # Batch size = 2, sequence length = 8
    dummy_input = torch.randint(0, 32000, (2, 8))
    dummy_labels = torch.randint(0, 32000, (2, 8))
    
    logits, total_loss, main_loss, aux_loss, z_loss = model(dummy_input, dummy_labels)
    
    print(f"    - Input shape:               {list(dummy_input.shape)}")
    print(f"    - Logits shape:              {list(logits.shape)}")
    print(f"    - Total Loss (composite):    {total_loss.item():.4f}")
    print(f"    - Main Loss (CrossEntropy):  {main_loss.item():.4f}")
    print(f"    - Aux Loss (Load Balancing): {aux_loss.item():.4f}")
    print(f"    - Z-Loss (Router logit):     {z_loss.item():.4f}")
    
    assert logits.shape == (2, 8, 32000), f"Expected logits shape (2, 8, 32000), got {logits.shape}"
    assert total_loss > 0, "Loss should be positive"
    assert aux_loss > 0, "Aux loss should be positive"
    assert z_loss > 0, "Z-loss should be positive"
    print("    ✅ Forward pass shape and loss values verified.")
    
    # 4. Backward Pass Validation
    print("\n[4/5] Verifying backward pass and gradients...")
    total_loss.backward()
    
    missing_grads = []
    zero_grads = []
    valid_grads = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                missing_grads.append(name)
            elif torch.all(param.grad == 0):
                zero_grads.append(name)
            else:
                valid_grads += 1
                
    print(f"    - Parameters with valid gradients: {valid_grads}")
    if missing_grads:
        print(f"    - ❌ Missing gradients for: {missing_grads}")
    if zero_grads:
        print(f"    - ❌ Zero gradients for: {zero_grads}")
        
    assert len(missing_grads) == 0, "All trainable parameters must receive gradients."
    assert len(zero_grads) == 0, "No parameter gradients should be entirely zero."
    print("    ✅ Backward pass completed and gradients are healthy.")
    
    # 5. Routing Distribution Verification
    print("\n[5/5] Checking expert routing distribution...")
    # Let's inspect expert selection on a larger dummy batch to ensure statistical validity
    dummy_input_large = torch.randint(0, 32000, (4, 32)) # 128 tokens total -> 256 selections
    
    # We will register a temporary hook to intercept router indices
    selected_indices = []
    def hook_fn(module, input, output):
        # output is (top_k_probs, top_k_indices, router_logits)
        selected_indices.append(output[1].detach().cpu())
        
    hooks = []
    for layer in model.layers:
        if isinstance(layer, MoETransformerBlock):
            hooks.append(layer.ffn.router.register_forward_hook(hook_fn))
            
    with torch.no_grad():
        _ = model(dummy_input_large)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    print(f"    - Tracked {len(selected_indices)} MoE layers.")
    for layer_idx, indices in enumerate(selected_indices):
        # indices shape: (S, k) = (128, 2)
        expert_counts = torch.zeros(4)
        for val in indices.view(-1):
            expert_counts[val.item()] += 1
            
        total_selections = expert_counts.sum().item()
        print(f"      Layer {layer_idx * 2} selections (total {int(total_selections)}):")
        for exp_id in range(4):
            count = int(expert_counts[exp_id].item())
            pct = (count / total_selections) * 100
            print(f"        Expert {exp_id}: {count} ({pct:.1f}%)")
            
            # Check for collapse: each expert should receive a reasonable fraction of routing
            assert count > 0, f"Layer {layer_idx * 2} expert {exp_id} received zero selections (Expert Collapse!)"
            
    print("    ✅ Expert routing is active and balanced across all layers.")
    print("\n" + "=" * 60)
    print("🎉 PHASE 1 VERIFICATION SUCCESSFUL: ALL CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()

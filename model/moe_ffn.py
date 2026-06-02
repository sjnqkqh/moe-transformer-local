import torch
import torch.nn as nn
from model.ffn import SwiGLU
from model.moe_router import MoERouter, load_balancing_loss, router_z_loss

class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        An individual expert FFN block using SwiGLU.
        """
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, k: int = 2):
        """
        Mixture of Experts FFN layer.
        
        Args:
            d_model (int): Model embedding dimension (768).
            d_ff (int): FFN hidden dimension (2048).
            num_experts (int): Total number of experts (4).
            k (int): Number of active experts per token (2).
        """
        super().__init__()
        self.router = MoERouter(d_model, num_experts, k=k)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.k = k

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input of shape (B, T, d_model).
            
        Returns:
            out (torch.Tensor): Output of shape (B, T, d_model).
            aux_loss (torch.Tensor): Load balancing loss (scalar).
            z_loss (torch.Tensor): Router z-loss (scalar).
        """
        B, T, C = x.shape
        flat_x = x.view(-1, C)  # Shape (S, d_model) where S = B * T
        
        # Get routing probabilities, selected expert indices, and raw logits
        top_k_probs, top_k_indices, router_logits = self.router(flat_x)
        
        # Calculate router losses
        aux_loss = load_balancing_loss(router_logits, top_k_indices)
        z_loss = router_z_loss(router_logits)
        
        # Initialize output tensor
        out = torch.zeros_like(flat_x)
        
        # Capacity-free routing: loop through each expert and process assigned tokens
        for i in range(self.num_experts):
            # Check which tokens have expert i in their top-k selections
            mask = (top_k_indices == i)  # (S, k)
            token_indices, k_indices = torch.where(mask)
            
            if len(token_indices) == 0:
                continue
            
            # Extract tokens routed to expert i
            selected_tokens = flat_x[token_indices]
            
            # Process tokens through expert i
            expert_out = self.experts[i](selected_tokens)
            
            # Scale by routing probability
            gating_weight = top_k_probs[token_indices, k_indices].unsqueeze(-1)
            scaled_out = expert_out * gating_weight
            
            # Scatter/accumulate results back to output tensor
            out.index_add_(0, token_indices, scaled_out)
            
        # Reshape output back to (B, T, d_model)
        out = out.view(B, T, C)
        return out, aux_loss, z_loss

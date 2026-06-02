import torch
import torch.nn as nn
from model.normalization import RMSNorm
from model.attention import MultiHeadAttention
from model.moe_ffn import MoEFFN

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int = 4, k: int = 2, max_seq_len: int = 1024, eps: float = 1e-6):
        """
        MoE Transformer layer block with pre-RMSNorm.
        
        Args:
            d_model (int): Model embedding dimension (768).
            n_heads (int): Number of attention heads (8).
            d_ff (int): Expert FFN hidden dimension (2048).
            num_experts (int): Total number of experts (4).
            k (int): Number of active experts per token (2).
            max_seq_len (int): Max sequence length for attention (1024).
            eps (float): Norm epsilon.
        """
        super().__init__()
        # Pre-RMSNorm for Attention
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len=max_seq_len)
        
        # Pre-RMSNorm for MoE FFN
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = MoEFFN(d_model, d_ff, num_experts=num_experts, k=k)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input of shape (B, T, d_model).
            freqs_cis (torch.Tensor): RoPE frequencies.
            
        Returns:
            x (torch.Tensor): Block output of shape (B, T, d_model).
            aux_loss (torch.Tensor): Router load balancing loss (scalar).
            z_loss (torch.Tensor): Router z-loss (scalar).
        """
        # Attention with residual
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        # MoE FFN with residual
        ffn_out, aux_loss, z_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, aux_loss, z_loss

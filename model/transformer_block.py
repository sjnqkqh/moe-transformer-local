import torch
import torch.nn as nn
from model.normalization import RMSNorm
from model.attention import MultiHeadAttention
from model.ffn import DenseFFN

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 1024, eps: float = 1e-6):
        """
        Standard Dense Transformer layer block with pre-RMSNorm.
        
        Args:
            d_model (int): Model embedding dimension (768).
            n_heads (int): Number of attention heads (8).
            d_ff (int): FFN hidden dimension (2048).
            max_seq_len (int): Max sequence length for attention (1024).
            eps (float): Norm epsilon.
        """
        super().__init__()
        # Pre-RMSNorm for Attention
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len=max_seq_len)
        
        # Pre-RMSNorm for FFN
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = DenseFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, T, d_model).
            freqs_cis (torch.Tensor): RoPE frequencies.
            
        Returns:
            torch.Tensor: Block output of shape (B, T, d_model).
        """
        # Attention with residual
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

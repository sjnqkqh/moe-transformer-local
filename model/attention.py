import math
import torch
import torch.nn as nn
from model.rope import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        """
        Causal Multi-Head Attention layer.
        
        Args:
            d_model (int): Dimension of the model (768).
            n_heads (int): Number of attention heads (8).
            max_seq_len (int): Maximum sequence length for causal masking (1024).
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # QKV Projections (bias=False is standard for modern Transformers)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        
        # Output Projection
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # Causal mask: upper triangle is -inf, lower triangle is 0
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            freqs_cis (torch.Tensor): Complex RoPE frequencies of shape (seq_len, head_dim // 2).
            
        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        xq = self.wq(x)  # (B, T, d_model)
        xk = self.wk(x)  # (B, T, d_model)
        xv = self.wv(x)  # (B, T, d_model)
        
        # Reshape for multi-head attention: (B, T, H, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_heads, self.head_dim)
        xv = xv.view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        # Transpose to (B, H, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Compute scaled attention scores: (B, H, T, T)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask: slice to current sequence length
        mask = self.mask[:T, :T]
        scores = scores + mask
        
        # Softmax to get probabilities
        probs = torch.softmax(scores, dim=-1)
        
        # Weighted sum of values: (B, H, T, head_dim)
        output = torch.matmul(probs, xv)
        
        # Re-assemble heads: (B, T, d_model)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        return self.wo(output)

import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE.
    
    Args:
        dim (int): Dimension of each attention head (head_dim). Must be even.
        end (int): Maximum sequence length (context_window).
        theta (float): Base for frequency calculation.
        
    Returns:
        torch.Tensor: Complex frequency tensor of shape (end, dim // 2).
    """
    assert dim % 2 == 0, "head_dim must be even for RoPE"
    
    # freqs = 1.0 / (theta ** (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # t = [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Outer product: shape (end, dim // 2)
    freqs = torch.outer(t, freqs).float()
    
    # Return polar coordinates representing e^(i * freqs)
    # torch.polar(abs, angle) -> abs * (cos(angle) + i * sin(angle))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape freqs_cis to broadcast with x.
    
    Args:
        freqs_cis (torch.Tensor): Complex tensor of shape (seq_len, head_dim // 2).
        x (torch.Tensor): Complex tensor of shape (batch_size, seq_len, num_heads, head_dim // 2).
        
    Returns:
        torch.Tensor: Reshaped tensor matching x's dimensions for broadcasting.
    """
    ndim = x.ndim
    assert ndim >= 2, "Tensor must have at least 2 dimensions"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis shape {freqs_cis.shape} must match x sequence length and head dimension {x.shape[1], x.shape[-1]}"
    
    # We want shape: (1, seq_len, 1, head_dim // 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings to query/key tensor.
    
    Args:
        x (torch.Tensor): Real tensor of shape (batch_size, seq_len, num_heads, head_dim).
        freqs_cis (torch.Tensor): Complex tensor of shape (seq_len, head_dim // 2).
        
    Returns:
        torch.Tensor: Rotated real tensor of shape (batch_size, seq_len, num_heads, head_dim).
    """
    # x shape: (B, T, H, head_dim) -> (B, T, H, head_dim // 2, 2)
    x_shaped = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    x_complex = torch.view_as_complex(x_shaped)
    
    # Align device of freqs_cis to x
    freqs_cis_device = freqs_cis.to(x.device)
    freqs_cis_broadcasted = reshape_for_broadcast(freqs_cis_device, x_complex)
    
    # Complex multiplication rotates coordinates
    rotated_complex = x_complex * freqs_cis_broadcasted
    
    # Convert back to real and flatten back to head_dim
    # rotated_complex shape: (B, T, H, head_dim // 2) -> (B, T, H, head_dim // 2, 2) -> (B, T, H, head_dim)
    rotated_real = torch.view_as_real(rotated_complex).contiguous()
    x_out = rotated_real.flatten(-2)
    
    return x_out.type_as(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Swish-gated Linear Unit (SwiGLU) Feed-Forward Network.
        
        Args:
            d_model (int): Input/output dimension.
            d_ff (int): Intermediate dimension.
        """
        super().__init__()
        # Gate path (w1) and value path (w2)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        # Output projection (w3)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU formula: (SiLU(x @ W1) * (x @ W2)) @ W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Dense FFN layer wrapper using SwiGLU.
        """
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

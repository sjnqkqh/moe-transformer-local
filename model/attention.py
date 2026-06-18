import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import apply_rotary_emb


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, max_seq_len: int = 1024, dropout: float = 0.0
    ):
        """
        인과적 멀티헤드 어텐션 (Causal Multi-Head Attention) 레이어.

        PyTorch 2.0+ SDPA(Scaled Dot-Product Attention)를 사용합니다.
        CUDA + BF16/FP16 환경(A100 등)에서 FlashAttention-2 커널이 자동으로 활성화됩니다.
        외부 flash-attn 패키지 설치 없이 동일한 성능을 제공합니다.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model은 n_heads의 배수여야 합니다."
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout  # SDPA에 직접 전달하기 위해 float로 보관

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.resid_dropout = nn.Dropout(p=dropout)
        # causal mask 버퍼 불필요 — SDPA의 is_causal=True가 대신 처리

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x         : (B, T, d_model)
            freqs_cos : (T, head_dim // 2) — RoPE cos(θ) 텐서
            freqs_sin : (T, head_dim // 2) — RoPE sin(θ) 텐서

        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape

        # [1] Q, K, V 투사
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # [2] 멀티헤드 분할: (B, T, H, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_heads, self.head_dim)
        xv = xv.view(B, T, self.n_heads, self.head_dim)

        # [3] RoPE 위치 인코딩
        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        # [4] SDPA 입력 형태로 변환: (B, H, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # [5] SDPA — causal masking + scaling + softmax + dropout 한 번에 처리
        #   is_causal=True  : 자동 인과 마스크 (수동 mask 버퍼 불필요)
        #   dropout_p       : 추론 시 자동으로 0으로 처리됨
        #   A100 BF16/FP16  : FlashAttention-2 커널 자동 선택
        dropout_p = self.dropout_p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # [6] 헤드 합치기: (B, H, T, head_dim) → (B, T, d_model)
        output = output.transpose(1, 2).contiguous().view(B, T, C)

        # [7] 출력 투사 + 잔차 드롭아웃
        return self.resid_dropout(self.wo(output))

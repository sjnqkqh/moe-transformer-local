from dataclasses import dataclass


@dataclass
class DenseTransformerConfig:
    """
    Dense Transformer 하이퍼파라미터.

    v8 스케일업 (~1.3B, Colab Pro+ A100 80GB 타깃):
      - 162M(d=768, 12L) → 1.3B(d=2048, 24L, 16H)
      - head_dim = 2048/16 = 128 (FlashAttention-2 친화, SDPA 자동 활성)
      - d_ff = 5632 (≈ d_model × 11/4, 64 배수)
      - 사전학습 dropout=0.0, SFT 단계에서 0.05로 상향 권장
      - eps=1e-5: 대형 모델 RMSNorm 안정성

    파라미터 추정: ~1.36B
      - emb(untied): 32000 × 2048 × 2 ≈ 131M
      - layer당: 4·2048² + 3·2048·5632 ≈ 51.4M
      - 24 layers: ~1.23B
    """

    vocab_size: int = 32000

    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 5632
    max_seq_len: int = 2048

    dropout: float = 0.0
    eps: float = 1e-5

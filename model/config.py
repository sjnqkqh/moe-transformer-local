from dataclasses import dataclass


@dataclass
class DenseTransformerConfig:
    """
    Dense Transformer 모델의 모든 하이퍼파라미터 설정을 한곳에 모아 관리하는 구성 클래스.
    """

    vocab_size: int = (
        32000  # BPE 토크나이저 어휘 사전 크기 (학습/평가 시 토크나이저 크기에 맞춰 동적 설정됨)
    )
    d_model: int = 768  # 토큰 임베딩 및 어텐션 은닉 차원 크기
    n_layers: int = 12  # 전체 레이어 층수 (12층 Dense)
    n_heads: int = 8  # 멀티헤드 어텐션 헤드 개수
    d_ff: int = 3072  # FFN (SwiGLU) 중간 은닉 차원 크기
    max_seq_len: int = 1024  # 최대 컨텍스트 윈도우 크기
    dropout: float = 0.1  # 어텐션 및 FFN에 적용할 드롭아웃 확률
    eps: float = 1e-6  # RMSNorm 수치 안정성을 위한 상수

from typing import Tuple

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE(Rotary Position Embedding)에 사용되는 회전 주파수의 cos/sin 값을 사전 계산합니다.

    torch.compile (Inductor) 완전 호환을 위해 복소수(complex) 텐서를 사용하지 않고,
    실수 cos/sin 텐서 쌍을 반환합니다.

    Args:
        dim (int): 각 어텐션 헤드의 차원 (head_dim). 반드시 짝수여야 합니다.
        end (int): 최대 허용 시퀀스 길이 (컨텍스트 윈도우 크기, 예: 2048).
        theta (float): 주파수 계산의 밑(base) 값. 표준 트랜스포머는 10000.0을 사용합니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - freqs_cos: cos(θ) 텐서, 형태 (end, dim // 2)
            - freqs_sin: sin(θ) 텐서, 형태 (end, dim // 2)
    """
    assert dim % 2 == 0, "RoPE를 적용하려면 head_dim이 짝수여야 합니다."

    # [과정 1] 헤드 차원의 절반에 대한 주파수(각속도) 구하기
    # freqs = 1.0 / (theta ** (2i / dim))
    # 각 인덱스마다 라디안 각도의 변화율이 다르게 배정됩니다.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # [과정 2] 시퀀스의 시간축 인덱스 t 생성: [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # [과정 3] 외적(Outer product)을 통해 시간 t와 각속도 freqs를 곱해 시간별 회전 각도 행렬 생성
    # 출력 형태: (end, dim // 2)
    freqs = torch.outer(t, freqs).float()

    # [과정 4] cos/sin 실수 텐서 직접 계산 (복소수 torch.polar 대신)
    # 크기가 1.0인 단위원 위의 점: cos(θ) + i·sin(θ) 에서 실수부/허수부만 추출
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    """
    Query 또는 Key 텐서에 RoPE 회전 변환을 실행하여 상대 위치 정보를 주입합니다.

    torch.compile (Inductor) 완전 호환: 복소수 텐서를 일절 사용하지 않습니다.
    수학적으로 기존 복소수 곱셈 버전과 완전히 동일합니다:
        (a + ib)(cosθ + isinθ) = (a·cosθ - b·sinθ) + i(a·sinθ + b·cosθ)

    Args:
        x (torch.Tensor): 어텐션 Q/K 실수 텐서. 형태: (batch_size, seq_len, num_heads, head_dim).
        freqs_cos (torch.Tensor): cos(θ) 텐서. 형태: (seq_len, head_dim // 2).
        freqs_sin (torch.Tensor): sin(θ) 텐서. 형태: (seq_len, head_dim // 2).

    Returns:
        torch.Tensor: 위치 회전이 적용된 실수 텐서. 형태: (batch_size, seq_len, num_heads, head_dim).
    """
    # [과정 1] cos/sin을 입력 dtype으로 변환 후 브로드캐스트 준비
    #   형태: (T, head_dim // 2) → (1, T, 1, head_dim // 2)
    cos = freqs_cos.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(2)

    # [과정 2] 인접한 짝수/홀수 채널 쌍 분리 — 2D 평면의 (a, b) 좌표에 대응
    x1 = x[..., 0::2]  # (B, T, H, head_dim // 2) — 짝수 인덱스
    x2 = x[..., 1::2]  # (B, T, H, head_dim // 2) — 홀수 인덱스

    # [과정 3] 2D 회전 적용 (실수 연산만 사용)
    #   out_even = a·cosθ - b·sinθ
    #   out_odd  = a·sinθ + b·cosθ
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # [과정 4] 짝수/홀수 결과를 인터리브하여 원래 head_dim 차원으로 합치기
    #   stack → (B, T, H, head_dim // 2, 2) → flatten → (B, T, H, head_dim)
    return torch.stack([out1, out2], dim=-1).flatten(-2)


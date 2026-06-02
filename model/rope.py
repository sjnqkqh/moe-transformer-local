import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    RoPE(Rotary Position Embedding)에 사용되는 회전 주파수 행렬을 복소수(complex) 형태로 사전 계산합니다.
    
    Args:
        dim (int): 각 어텐션 헤드의 차원 (head_dim). 반드시 짝수여야 합니다.
        end (int): 최대 허용 시퀀스 길이 (컨텍스트 윈도우 크기, 예: 1024).
        theta (float): 주파수 계산의 밑(base) 값. 표준 트랜스포머는 10000.0을 사용합니다.
        
    Returns:
        torch.Tensor: 복소수 형태의 회전 주파수 텐서. 형태: (end, dim // 2).
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
    
    # [과정 4] 극좌표계(Polar Coordinates)로 각 회전 각도를 복소수 평면상의 점 e^(i * theta)로 매핑합니다.
    # torch.polar(abs, angle) -> abs * (cos(angle) + i * sin(angle))
    # 크기(abs)가 1.0이므로 회전만 시키고 벡터의 L2 노름 크기는 보존됩니다.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    사전 계산된 주파수 텐서(freqs_cis)를 어텐션 입력 텐서(x)와 원활하게 원소별 곱셈(broadcasting)이 
    가능하도록 차원을 맞춰 재구성(reshape)합니다.
    
    Args:
        freqs_cis (torch.Tensor): 복소수 주파수 텐서. 형태: (seq_len, head_dim // 2).
        x (torch.Tensor): 어텐션 헤드 연산용 입력 텐서. 형태: (batch_size, seq_len, num_heads, head_dim // 2).
        
    Returns:
        torch.Tensor: 브로드캐스팅용으로 정렬된 텐서. 형태: (1, seq_len, 1, head_dim // 2).
    """
    ndim = x.ndim
    assert ndim >= 2, "입력 텐서는 2차원 이상이어야 합니다."
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"주파수 텐서의 형태 {freqs_cis.shape}가 입력의 시퀀스 길이 및 헤드 복소차원 {x.shape[1], x.shape[-1]}과 일치해야 합니다."
    
    # x의 배치 차원(0번째)과 헤드 차원(2번째) 자리에 1차원을 끼워 넣어 브로드캐스팅 형태를 만듭니다.
    # shape: (1, seq_len, 1, head_dim // 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Query 또는 Key 텐서에 RoPE 회전 변환을 실행하여 상대 위치 정보를 주입합니다.
    
    Args:
        x (torch.Tensor): 어텐션 Q/K 실제 실수 텐서. 형태: (batch_size, seq_len, num_heads, head_dim).
        freqs_cis (torch.Tensor): 사전 계산된 복소수 주파수 텐서. 형태: (seq_len, head_dim // 2).
        
    Returns:
        torch.Tensor: 위치 회전이 적용된 실수 텐서. 형태: (batch_size, seq_len, num_heads, head_dim).
    """
    # [과정 1] 인접한 2개 실수 쌍을 묶어 2D 평면상의 1개 점(복소수)으로 재배치합니다.
    # (B, T, H, head_dim) -> (B, T, H, head_dim // 2, 2)
    x_shaped = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    # (B, T, H, head_dim // 2) 복소수 텐서로 변환
    x_complex = torch.view_as_complex(x_shaped)
    
    # [과정 2] 장치(Device, CPU/MPS/CUDA)를 맞춰주고 브로드캐스트 차원 정렬
    freqs_cis_device = freqs_cis.to(x.device)
    freqs_cis_broadcasted = reshape_for_broadcast(freqs_cis_device, x_complex)
    
    # [과정 3] 복소수 곱셈 연산: (a + ib) * (cos + i sin) = (a cos - b sin) + i(a sin + b cos)
    # 2D 평면상의 좌표가 각 헤드 채널 쌍마다 상대 위치에 따라 각기 다른 각도로 회전하게 됩니다.
    rotated_complex = x_complex * freqs_cis_broadcasted
    
    # [과정 4] 복소수 결과를 다시 2차원 실수(real) 형태로 환원하고 원래의 헤드 차원으로 평탄화합니다.
    # (B, T, H, head_dim // 2) -> (B, T, H, head_dim // 2, 2) -> (B, T, H, head_dim)
    rotated_real = torch.view_as_real(rotated_complex).contiguous()
    x_out = rotated_real.flatten(-2)
    
    return x_out.type_as(x)

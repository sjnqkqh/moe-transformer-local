import math
import torch
import torch.nn as nn
from model.rope import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        """
        인과적 멀티헤드 어텐션 (Causal Multi-Head Attention) 레이어.
        디코더 전용 모델이므로 이전 단어만 참고하도록 마스킹(Causal Masking)을 수행합니다.
        
        Args:
            d_model (int): 모델 임베딩 차원 (768).
            n_heads (int): 어텐션 헤드 개수 (8).
            max_seq_len (int): 최대 학습 가능 시퀀스 길이 (1024).
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model은 n_heads의 배수여야 합니다."
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads # 헤드당 차원수: 768 / 8 = 96
        
        # Q, K, V 선형 레이어 정의 (bias=False는 학습이 과도하게 편향되는 것을 막아 줍니다.)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        
        # 출력 투사 레이어
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # 인과적 마스크 사전 등록 (Causal Mask)
        # 상삼각 행렬 위치(대각선 윗부분)는 -inf로 채우고, 대각선 포함 아랫부분은 0.0으로 둡니다.
        # 이렇게 하면 소프트맥스 직전에 점수 행렬과 더했을 때, 미래 단어 위치 점수가 -inf가 되어 확률이 0%가 됩니다.
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # register_buffer로 등록하면 가중치 파라미터가 아니므로 미분되지는 않으나, model.to(device) 호출 시 함께 이동합니다.
        # persistent=False는 체크포인트 파일(.pt) 저장 시 이 마스크 상수를 세이브하지 않도록 처리합니다.
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 시퀀스 텐서. 형태: (batch_size, seq_len, d_model)
            freqs_cis (torch.Tensor): 복소수 RoPE 주파수 텐서. 형태: (seq_len, head_dim // 2)
            
        Returns:
            torch.Tensor: 셀프 어텐션 수행 결과 텐서. 형태: (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape
        
        # [과정 1] Q, K, V 벡터 투사
        xq = self.wq(x)  # (B, T, d_model)
        xk = self.wk(x)  # (B, T, d_model)
        xv = self.wv(x)  # (B, T, d_model)
        
        # [과정 2] 멀티헤드 분할: (B, T, H, head_dim) 형태로 쪼개어 독립적인 어텐션 영역을 확보합니다.
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_heads, self.head_dim)
        xv = xv.view(B, T, self.n_heads, self.head_dim)
        
        # [과정 3] RoPE 위치 인코딩 반영
        # 슬라이싱 freqs_cis[:T]: 현재 인코딩할 텍스트 길이 T에 맞추어 주파수 배열을 슬라이싱해 입력합니다.
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        # [과정 4] 행렬곱 연산을 위해 차원 위치를 변경: (B, H, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # [과정 5] 어텐션 유사도 점수 계산 및 스케일링
        # (B, H, T, head_dim) @ (B, H, head_dim, T) -> (B, H, T, T)
        # sqrt(head_dim)으로 나눠주어 차원이 클 때 소프트맥스 값이 극단화되어 그래디언트가 소실되는 것을 제어합니다.
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # [과정 6] 인과적 마스킹 (Causal Masking) 적용
        # 현재 문장 길이 T에 맞춰 마스크 크기를 잘라 점수 행렬에 더합니다 (-inf 더하기).
        mask = self.mask[:T, :T]
        scores = scores + mask
        
        # [과정 7] 소프트맥스를 거쳐 미래 정보를 차단한 가중치 분포 도출
        probs = torch.softmax(scores, dim=-1)
        
        # [과정 8] 가중치 분포와 Value 벡터 행렬곱 연산: (B, H, T, head_dim)
        output = torch.matmul(probs, xv)
        
        # [과정 9] 모든 헤드 합치기 (Concatenation)
        # 차원 복원: (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, d_model)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # [과정 10] 최종 출력 투사 실행
        return self.wo(output)

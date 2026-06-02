import torch
import torch.nn as nn
from model.normalization import RMSNorm
from model.attention import MultiHeadAttention
from model.ffn import DenseFFN

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 1024, eps: float = 1e-6):
        """
        홀수 층에 사용되는 표준 밀집(Dense) Transformer 디코더 블록.
        프리 RMSNorm(Pre-RMSNorm) 레이아웃을 사용해 레이어가 매우 깊어져도 그래디언트가 우수하게 전파됩니다.
        
        Args:
            d_model (int): 모델 임베딩 차원 (768).
            n_heads (int): 멀티헤드 어텐션 헤드 개수 (8).
            d_ff (int): SwiGLU FFN 중간 숨은 차원 (2048).
            max_seq_len (int): 최대 컨텍스트 윈도우 크기 (1024).
            eps (float): RMSNorm 수치 안정성을 위한 작은 상수.
        """
        super().__init__()
        # 1. 셀프 어텐션 수행 전 입력 정규화를 수행할 RMSNorm
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len=max_seq_len)
        
        # 2. 피드포워드(FFN) 수행 전 입력 정규화를 수행할 RMSNorm
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = DenseFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서. 형태: (B, T, d_model)
            freqs_cis (torch.Tensor): 복소수 RoPE 주파수 버퍼.
            
        Returns:
            torch.Tensor: 레이어 출력을 더해준 잔차 결과 텐서. 형태: (B, T, d_model)
        """
        # [과정 1] Pre-RMSNorm Attention 적용 및 잔차 연결 (Residual Connection)
        # 입력을 RMSNorm 정규화한 뒤 어텐션을 통과시키고, 가중치 업데이트가 누락되지 않도록 원래 입력 x를 더해 줍니다.
        # x = x + Attention(RMSNorm(x))
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        
        # [과정 2] Pre-RMSNorm Dense FFN 적용 및 잔차 연결
        # x = x + DenseFFN(RMSNorm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

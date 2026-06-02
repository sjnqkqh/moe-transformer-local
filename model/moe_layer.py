import torch
import torch.nn as nn
from model.normalization import RMSNorm
from model.attention import MultiHeadAttention
from model.moe_ffn import MoEFFN

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int = 4, k: int = 2, max_seq_len: int = 1024, eps: float = 1e-6, dropout: float = 0.0):
        """
        짝수 층(0, 2, 4, 6)에 사용되는 Mixture of Experts (MoE) Transformer 디코더 블록.
        어텐션 층은 Dense FFN 블록과 완전히 공유되지만, FFN 층 자리에 다중 전문가를 호출하는 MoEFFN을 사용합니다.
        
        Args:
            d_model (int): 모델 임베딩 차원 (768).
            n_heads (int): 멀티헤드 어텐션 헤드 개수 (8).
            d_ff (int): 전문가 FFN의 은닉 중간 차원 (2048).
            num_experts (int): 전문가 수 (4).
            k (int): 활성 전문가 수 (2).
            max_seq_len (int): 최대 컨텍스트 윈도우 크기 (1024).
            eps (float): RMSNorm용 상수.
            dropout (float): 드롭아웃 확률.
        """
        super().__init__()
        # Pre-RMSNorm Attention
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
        
        # Pre-RMSNorm MoE FFN
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = MoEFFN(d_model, d_ff, num_experts=num_experts, k=k, dropout=dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 입력 텐서. 형태: (B, T, d_model)
            freqs_cis (torch.Tensor): 복소수 RoPE 주파수 버퍼.
            
        Returns:
            x (torch.Tensor): 레이어 잔차 합산 결과 출력. 형태: (B, T, d_model)
            aux_loss (torch.Tensor): 라우팅 분배 균등 보조 손실.
            z_loss (torch.Tensor): 라우터 로짓 안정화 보조 손실.
        """
        # [과정 1] Pre-RMSNorm Attention 적용 및 잔차 누적 더하기
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        
        # [과정 2] Pre-RMSNorm MoE FFN 적용 및 잔차 누적 더하기
        # 라우터 손실값인 aux_loss와 z_loss를 획득하여 상위 모듈로 계속 토스해 올려보냅니다.
        ffn_out, aux_loss, z_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        
        return x, aux_loss, z_loss

import torch
import torch.nn as nn
from model.ffn import SwiGLU
from model.moe_router import MoERouter, load_balancing_loss, router_z_loss

class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        """
        개별 전문가(Expert) 블록. 
        구조적으로는 Dense 층의 FFN과 동일한 SwiGLU 블록입니다.
        """
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, k: int = 2, dropout: float = 0.0):
        """
        Mixture of Experts FFN 레이어.
        
        Args:
            d_model (int): 모델의 임베딩 차원 크기 (768).
            d_ff (int): SwiGLU 중간 히든 레이어 차원 크기 (2048).
            num_experts (int): 전문가 FFN의 총 개수 (4).
            k (int): 한 개의 토큰이 보낼 활성 전문가 수 (2).
            dropout (float): 최종 출력에 적용할 드롭아웃 확률.
        """
        super().__init__()
        # 내부 라우터 객체 생성
        self.router = MoERouter(d_model, num_experts, k=k)
        # 4개의 전문가 FFN 객체 리스트 생성 (개별 전문가 내부 드롭아웃은 0.0으로 고정하여 최종 병합 출력이 드롭아웃되도록 함)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout=0.0) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.k = k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 입력 시퀀스 텐서. 형태: (B, T, d_model)
            
        Returns:
            out (torch.Tensor): 전문가 연산 가중합 결과 텐서. 형태: (B, T, d_model)
            aux_loss (torch.Tensor): 전문가 고르게 쓰기 보조 손실값 (스칼라)
            z_loss (torch.Tensor): 라우터 로짓 안정화 보조 손실값 (스칼라)
        """
        B, T, C = x.shape
        # [과정 1] 배치(B)와 시퀀스 길이(T)를 평탄화(Flatten)하여 토큰 단위로 변환합니다.
        # 이진/동적 라우팅 연산은 시퀀스 구분이 없으므로, (S, d_model) 차원으로 변환해 처리하는 것이 효율적입니다.
        flat_x = x.view(-1, C)  # 형태: (S, d_model), 여기서 S = B * T
        
        # [과정 2] 토큰들을 라우팅하여 각 토큰이 선택한 Top-K 전문가 확률과 인덱스 번호를 구합니다.
        # top_k_probs: (S, k) - 각 토큰이 고른 k개 전문가 분배 확률
        # top_k_indices: (S, k) - 각 토큰이 고른 k개 전문가 번호
        top_k_probs, top_k_indices, router_logits = self.router(flat_x)
        
        # [과정 3] 라우팅 균등 분배 손실 함수(Aux Loss) 및 Z-손실 계산
        aux_loss = load_balancing_loss(router_logits, top_k_indices)
        z_loss = router_z_loss(router_logits)
        
        # [과정 4] 최종 출력을 모아둘 텐서를 영(0)으로 초기화합니다.
        # 형태: (S, d_model)
        out = torch.zeros_like(flat_x)
        
        # [과정 5] 용량 제한 없는 라우팅 (Capacity-Free Routing) 루프
        # 모든 전문가(0번부터 3번까지)를 하나씩 순회하며, 해당 전문가에 할당된 토큰들만 골라 모아서 처리합니다.
        for i in range(self.num_experts):
            # top_k_indices 텐서에서 전문가 번호 i를 선택한 토큰의 위치 마스크를 만듭니다.
            # mask 형태: (S, k)
            mask = (top_k_indices == i)
            
            # mask가 True인 곳의 행(S차원 인덱스)하고 열(k차원 인덱스, 0 또는 1) 위치를 추출합니다.
            # token_indices: 해당 전문가를 고른 원래 토큰의 배치 위치 (예: [0, 5, 23, ...])
            # k_indices: 해당 토큰에서 몇 번째로(1등 혹은 2등) 그 전문가를 골랐는지 정보 (예: [0, 1, 0, ...])
            token_indices, k_indices = torch.where(mask)
            
            # 이번 전문가 i를 부른 토큰이 단 한 개도 없다면 건너뜁니다.
            if len(token_indices) == 0:
                continue
            
            # 전문가 i에게 가야 할 토큰들만 쏙 뽑아냅니다. (Gather 단계)
            # selected_tokens 형태: (N_selected, d_model)
            selected_tokens = flat_x[token_indices]
            
            # 해당 토큰들을 전문가 FFN에 통과시켜 계산합니다.
            expert_out = self.experts[i](selected_tokens)
            
            # 각 토큰이 이 전문가를 선택했을 때의 확률값(가중치)을 추출하고 곱해줍니다.
            # top_k_probs에서 각 토큰에 해당하는 위치의 값을 꺼내 가중치로 활용합니다.
            gating_weight = top_k_probs[token_indices, k_indices].unsqueeze(-1)
            scaled_out = expert_out * gating_weight
            
            # 연산 결과를 최종 출력 텐서의 원래 위치에 더해서 모아줍니다. (Scatter/Accumulate 단계)
            # index_add_(dim=0, index=token_indices, tensor=scaled_out)
            # 이 함수는 중복되지 않거나 중복되더라도 안전하게 값을 누적 더해줍니다.
            out.index_add_(0, token_indices, scaled_out)
            
        # [과정 6] 평탄화된 토큰 텐서를 원래 배치 형태로 복구시키고 최종 드롭아웃을 적용합니다.
        # (S, d_model) -> (B, T, d_model)
        out = out.view(B, T, C)
        return self.dropout(out), aux_loss, z_loss

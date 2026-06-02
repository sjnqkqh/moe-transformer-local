import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        """
        Mixture of Experts (MoE)를 위한 Top-K 라우터 모듈.
        
        Args:
            d_model (int): 모델의 임베딩 차원 크기 (예: 768).
            num_experts (int): 전체 전문가(Expert) FFN의 개수 (예: 4).
            k (int): 한 개의 토큰이 도달할 활성화 전문가 개수 (예: 2).
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        # 게이팅 레이어: 토큰 임베딩 벡터를 입력받아 각 전문가에 대한 비정규화 선호도(logits)를 출력합니다.
        # 편향(bias)을 두지 않는 것이 표준입니다. bias가 있으면 입력 값과 상관없이 특정 전문가만 선호하는 편향이 생길 수 있기 때문입니다.
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 평탄화된 토큰 벡터 모음. 형태: (S, d_model), 여기서 S = 배치 크기 * 시퀀스 길이.
                              
        Returns:
            top_k_probs (torch.Tensor): 선택된 Top-K 전문가들에게 분배할 게이팅 확률값. 형태: (S, k).
            top_k_indices (torch.Tensor): 선택된 Top-K 전문가들의 인덱스 번호. 형태: (S, k).
            router_logits (torch.Tensor): 소프트맥스를 취하기 전의 원시 게이팅 로짓 값. 형태: (S, num_experts).
        """
        # [과정 1] 각 토큰마다 전문가 선호도 로짓 계산: (S, num_experts)
        router_logits = self.gate(x)
        
        # [과정 2] 소프트맥스를 취하여 전문가 선택 확률 분포 생성: (S, num_experts)
        probs = F.softmax(router_logits, dim=-1)
        
        # [과정 3] 가장 확률이 높은 상위 k개의 전문가와 해당 확률값 추출
        # top_k_probs: (S, k) - 선택된 전문가들의 소프트맥스 확률값
        # top_k_indices: (S, k) - 선택된 전문가들의 인덱스 (0, 1, 2, 3 중 하나)
        top_k_probs, top_k_indices = torch.topk(probs, k=self.k, dim=-1)
        
        # [과정 4] 선택된 k개 전문가의 확률 합이 1이 되도록 재정규화 (Renormalization)
        # 예: 원래 소프트맥스 확률이 [0.4, 0.3, 0.2, 0.1]이고 k=2이면 [0.4, 0.3]을 뽑은 후,
        # 각각 0.4/(0.4+0.3) = 0.57, 0.3/(0.4+0.3) = 0.43으로 보정하여 최종 출력 가중치로 사용합니다.
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices, router_logits

def load_balancing_loss(router_logits: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
    """
    모든 전문가가 균등한 빈도로 선택되도록 강제하는 보조 손실 함수 (Shazeer et al., 2017).
    특정 전문가 한두 개로 토큰이 쏠리는 'Expert Collapse'를 방지합니다.
    
    연산 공식: E * sum(f_e * P_e)
      - E: 전문가 개수
      - f_e: 전문가 e가 Top-K에 선택된 실제 토큰 비율 (Hard routing ratio)
      - P_e: 전문가 e에 할당된 라우터 소프트맥스 평균 확률 (Soft routing probability)
      
    Args:
        router_logits (torch.Tensor): 라우터 출력 로짓. 형태: (S, num_experts).
        top_k_indices (torch.Tensor): 선택된 Top-K 전문가 인덱스. 형태: (S, k).
        
    Returns:
        torch.Tensor: 로드 밸런싱 손실 (스칼라 값, 최적화 시 0.01 계수를 곱해 메인 손실과 합산).
    """
    S, E = router_logits.shape
    probs = F.softmax(router_logits, dim=-1)
    
    # [과정 1] 이진 마스크 생성: 각 토큰이 선택한 Top-K 위치에 1.0을 표시합니다.
    # shape: (S, E)
    mask = torch.zeros_like(probs)
    mask.scatter_(1, top_k_indices, 1.0)
    
    # [과정 2] f_e 계산: 전체 토큰 중 전문가 e를 선택한 비율을 계산합니다 (평균 연산).
    # shape: (E,)
    f = mask.mean(dim=0)
    
    # [과정 3] P_e 계산: 모든 토큰에서 전문가 e가 받은 소프트맥스 확률값의 평균을 계산합니다.
    # shape: (E,)
    P = probs.mean(dim=0)
    
    # [과정 4] 최종 손실 연산
    # 완전히 균등 분배될 때 (f_e = k/E, P_e = 1/E) 최소치(k)를 가집니다.
    # 미분 불가능한 mask 대신, probs에서 파생된 P가 그래디언트 역전파의 통로 역할을 합니다.
    aux_loss = E * torch.sum(f * P)
    return aux_loss

def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    라우터 로짓의 절대 크기가 너무 커지는 현상(Explosion)을 제어하는 라우터 Z-손실 함수 (ST-MoE 기법).
    로그 소프트맥스의 분모(Partition Function)를 직접 억제하여 FP16/BF16 연산 중 오버플로우를 차단합니다.
    
    연산 공식: mean( log(sum(exp(router_logits_i)))^2 )
    
    Args:
        router_logits (torch.Tensor): 라우터 출력 로짓. 형태: (S, num_experts).
        
    Returns:
        torch.Tensor: 라우터 Z-손실 (스칼라 값, 최적화 시 0.001 계수를 곱해 메인 손실과 합산).
    """
    # log(sum(exp(logits)))를 계산합니다 (logsumexp로 안전하게 연산)
    log_z = torch.logsumexp(router_logits, dim=-1)
    # 분모의 제곱값을 평균 내어 로짓 크기가 폭발하는 것을 페널티 항으로 잡습니다.
    return torch.mean(log_z ** 2)

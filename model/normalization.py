import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization (RMSNorm).
        표준 LayerNorm 대비 평균(Mean)을 빼는 연산을 생략하여 계산 비용을 약 7% ~ 10% 절감하는 경량 노멀라이제이션.
        Gemma, LLaMA 등 현대 거대 언어 모델(LLM)들의 표준 정규화 기법입니다.

        Args:
            dim (int): 모델의 임베딩 차원 (d_model, 예: 768).
            eps (float): 0분모 나눗셈 방지를 위한 아주 작은 스칼라 상수 (1e-6).
        """
        super().__init__()
        self.eps = eps
        # 정규화된 텐서에 곱해줄 학습 가능한 가중치(Scale parameter) 정의 (1.0으로 초기화)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        주어진 텐서 x를 L2 노름 평균 제곱근(RMS)으로 나누어 정규화합니다.
        공식: x / sqrt(mean(x^2, dim=-1) + eps)
        """
        # x.pow(2).mean(-1, keepdim=True): 마지막 차원(차원값 768)에 대해 원소 제곱 평균을 구합니다.
        # rsqrt: 제곱근의 역수(1 / sqrt(x))를 고속 계산합니다.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서. 형태: (Batch, Seq_Len, d_model)

        Returns:
            torch.Tensor: RMS 정규화 및 크기 보정(Weight 곱)이 완료된 텐서.
        """
        # BF16/FP16 혼합 정밀도 학습 중 언더플로우/오버플로우를 막고 수치적 안정성을 확보하기 위해
        # 분모 제곱 평균 연산은 float32 정밀도로 형변환하여 수행한 뒤, 다시 원래 입력 타입(BF16 등)으로 복구합니다.
        output = self._norm(x.float()).type_as(x)
        # 학습 가능한 가중치 weight를 원소별로 곱하여 최종 스케일을 조정합니다.
        return output * self.weight

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Swish-gated Linear Unit (SwiGLU) 피드포워드 네트워크.
        GELU 기반 FFN 대비 언어 모델 당혹도(Perplexity)를 1% ~ 2% 낮추어 주어,
        현대 대부분의 LLM(LLaMA, PaLM 등)에서 채택한 FFN 레이아웃입니다.
        
        Args:
            d_model (int): 입력 및 출력 벡터 차원 크기 (768).
            d_ff (int): 내부 은닉 차원 크기 (2048).
                        (SwiGLU는 두 번의 투사를 수행하므로, 파라미터 총량을 Dense FFN과 맞추기 위해 보통 8/3배 보정을 적용합니다.)
        """
        super().__init__()
        # 게이트 경로 (Gate path): 입력을 히든 차원으로 매핑 후 SiLU(Swish) 활성화를 통과시킬 가중치
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # 값 경로 (Value path): 게이트 가중치와 곱해져 값을 조율할 가중치
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        # 출력 복구 경로: 차원을 다시 원래 임베딩 차원(d_model)으로 축소 투사할 가중치
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서. 형태: (Batch, Seq_Len, d_model)
            
        Returns:
            torch.Tensor: SwiGLU 연산이 끝난 출력 텐서. 형태: (Batch, Seq_Len, d_model)
        """
        # SwiGLU 공식: (SiLU(x @ W1) * (x @ W2)) @ W3
        # F.silu(self.w1(x)): 게이트 경로에 SiLU(Swish) 함수를 적용
        # self.w2(x): 값 경로에 선형 투사 수행
        # * 연산: 두 경로 출력 텐서의 원소별 곱셈 (Element-wise multiplication)
        # self.w3(...): 최종 차원 복사 투사
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        홀수 레이어에서 사용되는 단일 밀집(Dense) 피드포워드 네트워크 래퍼 모듈.
        """
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

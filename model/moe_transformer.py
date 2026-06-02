import torch
import torch.nn as nn
import torch.nn.functional as F
from model.normalization import RMSNorm
from model.rope import precompute_freqs_cis
from model.transformer_block import TransformerBlock
from model.moe_layer import MoETransformerBlock
from model.config import MoETransformerConfig

class MoETransformer(nn.Module):
    def __init__(self, config: MoETransformerConfig = None, **kwargs):
        """
        교차 배치 구조의 Decoder-only Mixture of Experts (MoE) Transformer 전체 모델 클래스.
        
        Args:
            config (MoETransformerConfig, optional): 모델의 아키텍처 설정을 담은 객체.
            **kwargs: config가 없을 때 개별적으로 전달할 수 있는 하이퍼파라미터 인자 (하위 호환성용).
        """
        super().__init__()
        if config is None:
            config = MoETransformerConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.max_seq_len = config.max_seq_len
        eps = config.eps
        
        # [구조 1] 단어 토큰 임베딩 선형 행렬 (가중치 미공유 untied 구조)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        
        # [구조 2] 교차 배치 블록 리스트 생성
        # 짝수 레이어(0, 2, 4, 6) = MoETransformerBlock (전문가 4개 활성화)
        # 홀수 레이어(1, 3, 5, 7) = TransformerBlock (Dense 밀집 레이어)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i % 2 == 0:
                self.layers.append(MoETransformerBlock(
                    d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
                    num_experts=config.num_experts, k=config.k, max_seq_len=self.max_seq_len, eps=eps,
                    dropout=config.dropout
                ))
            else:
                self.layers.append(TransformerBlock(
                    d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
                    max_seq_len=self.max_seq_len, eps=eps,
                    dropout=config.dropout
                ))
                
        # [구조 3] 최종 RMSNorm 레이어
        # Pre-RMSNorm 구조이므로 최종 분류(Linear) 단계 진입 직전에 정규화 처리가 수반되어야 합니다.
        self.norm = RMSNorm(self.d_model, eps=eps)
        
        # [구조 4] 최종 어휘 사전 매핑 분류기 (LM Head, Untied 구조)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # [구조 5] RoPE 회전 주파수 극좌표 복소 버퍼 생성 및 등록
        freqs_cis = precompute_freqs_cis(self.d_model // self.n_heads, self.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            input_ids (torch.Tensor): 토큰 인덱스 행렬. 형태: (batch_size, seq_len)
            labels (torch.Tensor, optional): 정답 타겟 토큰 인덱스. 형태: (batch_size, seq_len)
            
        Returns:
            logits (torch.Tensor): 다음 단어 예측 점수 분포. 형태: (batch_size, seq_len, vocab_size)
            loss (torch.Tensor, optional): 최종 복합 학습 손실 값 (Labels 존재 시).
            main_loss (torch.Tensor, optional): 크로스엔트로피 어휘 학습 손실 값 (Labels 존재 시).
            total_aux_loss (torch.Tensor, optional): 라우터 로드밸런싱 손실의 전체 레이어 합산 값.
            total_z_loss (torch.Tensor, optional): 라우터 Z-손실의 전체 레이어 합산 값.
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"입력 시퀀스 길이 {T}가 모델 허용 한도 {self.max_seq_len}를 초과했습니다."
        
        # 1단계: 토큰들을 벡터 차원으로 임베딩 변환
        x = self.token_embeddings(input_ids)
        
        # 2단계: 현재 시퀀스 길이 T에 맞춰 RoPE 주파수 버퍼 잘라오기
        freqs_cis = self.freqs_cis[:T]
        
        # 라우터 보조 손실 누적 수집기
        total_aux_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)
        
        # 3단계: 8개 교차 레이어 순전파 실행
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                # MoE 층은 연산 출력뿐만 아니라 라우터 손실들을 함께 리턴받아 누적합니다.
                x, aux_l, z_l = layer(x, freqs_cis)
                total_aux_loss = total_aux_loss + aux_l
                total_z_loss = total_z_loss + z_l
            else:
                # Dense 층은 어텐션과 단일 FFN만 단순 연산합니다.
                x = layer(x, freqs_cis)
                
        # 4단계: 최종 정규화 및 분류기를 통한 로짓(Logits) 변환
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        main_loss = None
        if labels is not None:
            # [Causal Shift 과정]
            # 인과적 언어 모델링(Causal LM) 학습을 위해 로짓과 라벨 위치를 각각 한 칸씩 밀어서 짝을 맞춰줍니다.
            # 시간 t 시점의 logits으로 시간 t+1 시점의 labels를 예측하도록 훈련시킵니다.
            shift_logits = logits[..., :-1, :].contiguous() # 마지막 시점 로짓 탈락
            shift_labels = labels[..., 1:].contiguous()     # 첫 시점 정답 탈락
            
            # 크로스엔트로피 손실 함수 계산 (ignore_index=-100은 패딩 단어 학습 연산 무시 설정)
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
            
            # [최종 복합 손실 공식]
            # total_loss = main_loss + 0.01 * aux_loss + 0.001 * z_loss
            # 보조 손실에 알맞은 페널티 계수를 곱해 합쳐줌으로써 라우터가 동시에 역전파 학습되도록 유도합니다.
            loss = main_loss + 0.01 * total_aux_loss + 0.001 * total_z_loss
            
        return logits, loss, main_loss, total_aux_loss, total_z_loss

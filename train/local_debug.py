import os
import sys

import torch

# 프로젝트 루트 경로를 시스템 경로에 등록하여 model 패키지를 참조할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig


def main():
    print("=" * 60)
    print("🤖 Dense Transformer — Local Debug Verification")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # [1단계] 모델 객체 인스턴스화 (Model Instantiation)
    # -------------------------------------------------------------------------
    # 청사진(Blueprint)에 정의된 사양대로 모델의 레이어 개수와 가중치 모양을 할당합니다.
    # 이 과정에서 내부적으로 RMSNorm, RoPE 주파수 버퍼, Attention, SwiGLU FFN들이 결합됩니다.
    print("\n[1/4] Instantiating model...")
    config = DenseTransformerConfig(
        vocab_size=32000,  # BPE 토크나이저 어휘 사전 크기
        d_model=768,  # 토큰 임베딩 차원 크기
        n_layers=12,  # 총 레이어 층수 (12층 Dense)
        n_heads=8,  # 멀티헤드 어텐션 헤드 개수
        d_ff=3072,  # SwiGLU FFN 중간 은닉 차원 크기
        max_seq_len=1024,  # 최대 컨텍스트 윈도우 크기
        dropout=0.1  # 기본 드롭아웃 확률 설정
    )
    model = DenseTransformer(config)
    print("    ✅ Model instantiated successfully.")

    # -------------------------------------------------------------------------
    # [2단계] 파라미터 수 정밀 검증 (Parameter Count Check)
    # -------------------------------------------------------------------------
    # 트랜스포머 모델의 전체 가중치 파라미터 개수가 설계서상의 파라미터 규모와 일치하는지 계산합니다.
    # 가중치가 공유되지 않는 untied 임베딩 구조이므로 총 파라미터는 약 162.4M 수준이 나와야 합니다.
    print("\n[2/4] Counting parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    emb_params = model.token_embeddings.weight.numel()
    attn_params = sum(
        p.numel() for name, p in model.named_parameters() if "attention" in name
    )
    ffn_params = sum(
        p.numel() for name, p in model.named_parameters() if "ffn" in name and "attention" not in name
    )
    lm_head_params = model.lm_head.weight.numel()

    print(f"    - Total Parameters:      {total_params:,} ({total_params / 1e6:.2f}M)")
    print(
        f"    - Trainable Parameters:  {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
    )
    print(f"    - Token Embedding:       {emb_params:,} ({emb_params / 1e6:.2f}M)")
    print(f"    - Attention (12 layers): {attn_params:,} ({attn_params / 1e6:.2f}M)")
    print(f"    - FFN (12 layers):       {ffn_params:,} ({ffn_params / 1e6:.2f}M)")
    print(f"    - LM Head (untied):      {lm_head_params:,} ({lm_head_params / 1e6:.2f}M)")

    # 가중치 크기가 정상 범주인지 assert로 최종 보장합니다.
    assert 160e6 < total_params < 165e6, (
        f"Expected parameters to be ~162.4M, got {total_params / 1e6:.2f}M"
    )
    print("    ✅ Parameter counts are within the expected range (~162.4M untied).")

    # -------------------------------------------------------------------------
    # [3단계] 순전파 차원 및 손실 검증 (Forward Pass & Loss Check)
    # -------------------------------------------------------------------------
    # 모델에 가상의 미니 배치 데이터를 먹여서 연산 오류 없이 logits를 출력하고 loss를 산출하는지 확인합니다.
    print("\n[3/4] Verifying forward pass with dummy batch...")
    # 배치 크기(B)=2, 시퀀스 길이(T)=8인 임의의 토큰 인덱스를 생성합니다. (어휘 사전 범위 0 ~ 31999)
    dummy_input = torch.randint(0, 32000, (2, 8))
    dummy_labels = torch.randint(0, 32000, (2, 8))
    # 순전파 실행: 손실 함수 계산을 포함합니다.
    logits, total_loss, main_loss = model(dummy_input, dummy_labels)

    print(f"    - Input shape:               {list(dummy_input.shape)}")
    print(f"    - Logits shape:              {list(logits.shape)}")
    print(f"    - Total Loss:                {total_loss.item():.4f}")
    print(f"    - Main Loss (CrossEntropy):  {main_loss.item():.4f}")

    # 토큰별 어휘 예측 확률 차원이 올바른지 검사합니다: (Batch, Seq_Len, Vocab_Size)
    assert logits.shape == (2, 8, 32000), (
        f"Expected logits shape (2, 8, 32000), got {logits.shape}"
    )
    assert total_loss > 0, "손실은 양수여야 합니다."
    print("    ✅ Forward pass shape and loss values verified.")

    # -------------------------------------------------------------------------
    # [4단계] 역전파 및 그래디언트 소실 검사 (Backward Pass & Gradient Check)
    # -------------------------------------------------------------------------
    # 역전파(Backpropagation) 연산을 수행한 뒤, 모든 학습 대상 가중치의 미분값(.grad)이 정상 주입되었는지 확인합니다.
    # 미분값이 None이거나 0.0이면 코드가 단절되었거나(detached), 데드 경로가 존재한다는 버그의 확실한 증거입니다.
    print("\n[4/4] Verifying backward pass and gradients...")
    total_loss.backward()

    missing_grads = []
    zero_grads = []
    valid_grads = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                # 미분 계산이 누락된 텐서 수집
                missing_grads.append(name)
            elif torch.all(param.grad == 0):
                # 미분값이 전부 0으로 수렴한 텐서 수집 (기울기 소실 버그)
                zero_grads.append(name)
            else:
                valid_grads += 1

    print(f"    - Parameters with valid gradients: {valid_grads}")
    if missing_grads:
        print(f"    - ❌ Missing gradients for: {missing_grads}")
    if zero_grads:
        print(f"    - ❌ Zero gradients for: {zero_grads}")

    assert len(missing_grads) == 0, (
        "모든 학습 대상 매개변수에는 미분값이 주입되어야 합니다."
    )
    assert len(zero_grads) == 0, (
        "미분값 피드백이 전부 0인 매개변수가 존재해서는 안 됩니다."
    )
    print("    ✅ Backward pass completed and gradients are healthy.")

    print("\n" + "=" * 60)
    print("🎉 PHASE 1 VERIFICATION SUCCESSFUL: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

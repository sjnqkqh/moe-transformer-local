import os
import sys

import torch

# 프로젝트 루트 경로를 시스템 경로에 등록하여 model 패키지를 참조할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.moe_layer import MoETransformerBlock
from model.moe_transformer import MoETransformer
from model.config import MoETransformerConfig


def main():
    print("=" * 60)
    print("🤖 MoE Transformer — Local Debug Verification")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # [1단계] 모델 객체 인스턴스화 (Model Instantiation)
    # -------------------------------------------------------------------------
    # 청사진(Blueprint)에 정의된 사양대로 모델의 레이어 개수와 가중치 모양을 할당합니다.
    # 이 과정에서 내부적으로 RMSNorm, RoPE 주파수 버퍼, Attention, SwiGLU FFN들이 결합됩니다.
    print("\n[1/5] Instantiating model...")
    config = MoETransformerConfig(
        vocab_size=32000,  # BPE 토크나이저 어휘 사전 크기
        d_model=768,  # 토큰 임베딩 차원 크기
        n_layers=8,  # 총 레이어 층수 (교차 배치)
        n_heads=8,  # 멀티헤드 실어텐션 헤드 개수
        d_ff=2048,  # SwiGLU FFN 중간 은닉 차원 크기
        num_experts=4,  # MoE 레이어당 전문가 개수
        k=2,  # Top-2 라우팅 활성화 수
        max_seq_len=1024,  # 최대 컨텍스트 윈도우 크기
        dropout=0.1  # 기본 드롭아웃 확률 설정
    )
    model = MoETransformer(config)
    print("    ✅ Model instantiated successfully.")

    # -------------------------------------------------------------------------
    # [2단계] 파라미터 수 정밀 검증 (Parameter Count Check)
    # -------------------------------------------------------------------------
    # 트랜스포머 모델의 전체 가중치 파라미터 개수가 설계서상의 파라미터 규모와 일치하는지 계산합니다.
    # 가중치가 공유되지 않는 untied 임베딩 구조이므로 총 파라미터는 약 162M 수준이 나와야 합니다.
    print("\n[2/5] Counting parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    emb_params = model.token_embeddings.weight.numel()
    attn_params = sum(
        p.numel() for name, p in model.named_parameters() if "attention" in name
    )

    # 레이어 인덱스가 짝수(0, 2, 4, 6)이면 MoE FFN으로 분류하고, 홀수(1, 3, 5, 7)이면 Dense FFN으로 분류합니다.
    dense_ffn_params = 0
    moe_ffn_params = 0
    for name, p in model.named_parameters():
        if "ffn" in name and "attention" not in name:
            parts = name.split(".")
            if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx % 2 == 0:
                    moe_ffn_params += p.numel()
                else:
                    dense_ffn_params += p.numel()

    lm_head_params = model.lm_head.weight.numel()

    print(f"    - Total Parameters:      {total_params:,} ({total_params / 1e6:.2f}M)")
    print(
        f"    - Trainable Parameters:  {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
    )
    print(f"    - Token Embedding:       {emb_params:,} ({emb_params / 1e6:.2f}M)")
    print(f"    - Attention (8 layers):  {attn_params:,} ({attn_params / 1e6:.2f}M)")
    print(
        f"    - Dense FFN (4 layers):  {dense_ffn_params:,} ({dense_ffn_params / 1e6:.2f}M)"
    )
    print(
        f"    - MoE FFN (4 layers):    {moe_ffn_params:,} ({moe_ffn_params / 1e6:.2f}M)"
    )
    print(
        f"    - LM Head (untied):      {lm_head_params:,} ({lm_head_params / 1e6:.2f}M)"
    )

    # 가중치 크기가 정상 범주인지 assert로 최종 보장합니다.
    assert 160e6 < total_params < 165e6, (
        f"Expected parameters to be ~162.4M, got {total_params / 1e6:.2f}M"
    )
    print("    ✅ Parameter counts are within the expected range (~162.4M untied).")

    # -------------------------------------------------------------------------
    # [3단계] 순전파 차원 및 복합 손실 검증 (Forward Pass & Loss Check)
    # -------------------------------------------------------------------------
    # 모델에 가상의 미니 배치 데이터를 먹여서 연산 오류 없이 logits를 출력하고 loss를 산출하는지 확인합니다.
    print("\n[3/5] Verifying forward pass with dummy batch...")
    # 배치 크기(B)=2, 시퀀스 길이(T)=8인 임의의 토큰 인덱스를 생성합니다. (어휘 사전 범위 0 ~ 31999)
    dummy_input = torch.randint(0, 32000, (2, 8))
    dummy_labels = torch.randint(0, 32000, (2, 8))
    # 순전파 실행: 복합 손실 함수 계산을 포함합니다.
    logits, total_loss, main_loss, aux_loss, z_loss = model(dummy_input, dummy_labels)

    print(f"    - Input shape:               {list(dummy_input.shape)}")
    print(f"    - Logits shape:              {list(logits.shape)}")
    print(f"    - Total Loss (composite):    {total_loss.item():.4f}")
    print(f"    - Main Loss (CrossEntropy):  {main_loss.item():.4f}")
    print(f"    - Aux Loss (Load Balancing): {aux_loss.item():.4f}")
    print(f"    - Z-Loss (Router logit):     {z_loss.item():.4f}")

    # 토큰별 어휘 예측 확률 차원이 올바른지 검사합니다: (Batch, Seq_Len, Vocab_Size)
    assert logits.shape == (2, 8, 32000), (
        f"Expected logits shape (2, 8, 32000), got {logits.shape}"
    )
    assert total_loss > 0, "손실은 양수여야 합니다."
    assert aux_loss > 0, "라우팅 활성화 비율에 의한 로드 밸런싱 손실은 존재해야 합니다."
    assert z_loss > 0, "라우팅 오버플로우 방지용 Z-손실은 존재해야 합니다."
    print("    ✅ Forward pass shape and loss values verified.")

    # -------------------------------------------------------------------------
    # [4단계] 역전파 및 그래디언트 소실 검사 (Backward Pass & Gradient Check)
    # -------------------------------------------------------------------------
    # 역전파(Backpropagation) 연산을 수행한 뒤, 모든 학습 대상 가중치의 미분값(.grad)이 정상 주입되었는지 확인합니다.
    # 미분값이 None이거나 0.0이면 코드가 단절되었거나(detached), 데드 경로가 존재한다는 버그의 확실한 증거입니다.
    print("\n[4/5] Verifying backward pass and gradients...")
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

    # -------------------------------------------------------------------------
    # [5단계] 라우터 포워드 훅을 이용한 전문가 골고루 쓰기 검증 (Routing Health)
    # -------------------------------------------------------------------------
    # 배치 크기를 확장해 128개의 토큰을 통과시키고, PyTorch의 forward_hook을 활용하여 각 MoE 레이어 내의
    # 라우터가 전문가 4개로 균형있게 분배해 보냈는지 카운팅하여 쏠림 현상(Collapse)을 수치적으로 증명합니다.
    print("\n[5/5] Checking expert routing distribution...")
    dummy_input_large = torch.randint(
        0, 32000, (4, 32)
    )  # 4 * 32 = 128 토큰 (Top-2이므로 총 256번의 선택)

    # 포워드 훅 함수: 라우터 연산 완료 직후 가로채서 전문가 번호 텐서만 복사해 저장합니다.
    selected_indices = []

    def hook_fn(module, input, output):
        # output[1] = top_k_indices (S, k)
        selected_indices.append(output[1].detach().cpu())

    hooks = []
    # 8개 레이어를 순회하며 MoE 블록의 라우터 포워드에 훅을 거치합니다.
    for layer in model.layers:
        if isinstance(layer, MoETransformerBlock):
            hooks.append(layer.ffn.router.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(dummy_input_large)

    # 메모리 누수 방지를 위해 설치한 훅을 완전히 해제합니다.
    for h in hooks:
        h.remove()

    print(f"    - Tracked {len(selected_indices)} MoE layers.")
    for layer_idx, indices in enumerate(selected_indices):
        # indices shape: (128, 2)
        num_experts = model.config.num_experts
        expert_counts = torch.bincount(indices.view(-1), minlength=num_experts).float()
        expert_counts = expert_counts[:num_experts]

        total_selections = expert_counts.sum().item()
        print(
            f"      Layer {layer_idx * 2} selections (total {int(total_selections)}):"
        )
        for exp_id in range(num_experts):
            count = int(expert_counts[exp_id].item())
            pct = (count / total_selections) * 100
            print(f"        Expert {exp_id}: {count} ({pct:.1f}%)")

            # 전문가 쏠림 체크: 4개 전문가 중 단 한 번도 호출되지 않은 낙오자가 있는지 체크합니다.
            assert count > 0, (
                f"Layer {layer_idx * 2} expert {exp_id} received zero selections (Expert Collapse!)"
            )

    print("    ✅ Expert routing is active and balanced across all layers.")
    print("\n" + "=" * 60)
    print("🎉 PHASE 1 VERIFICATION SUCCESSFUL: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

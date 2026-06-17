"""
verify_flash_attention.py — FlashAttention-2 백엔드 가능 여부 검증

학습 환경(A100 BF16 등)에서 SDPA가 FlashAttention-2 커널을 선택할 수 있는지
확인하는 sanity check. 학습 전 한 번 실행 권장.

실행:
    python train/verify_flash_attention.py

판정:
    ✅ Flash 가능 → 모델은 SDPA 자동 선택으로 충분 (메모리/속도 정상)
    ❌ Flash 실패 → mem_efficient/math 백엔드로 fallback,
                  model/attention.py에서 sdpa_kernel(FLASH_ATTENTION) 강제 필요
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def verify(
    batch: int = 8,
    n_heads: int = 16,
    seq_len: int = 2048,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    if not torch.cuda.is_available():
        print("❌ CUDA 없음 — GPU 환경에서 실행하세요.")
        return False

    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Test shape: B={batch}, H={n_heads}, T={seq_len}, "
        f"D={head_dim}, dtype={dtype}"
    )

    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        print(f"✅ FlashAttention-2 사용 가능 — 출력 shape: {tuple(out.shape)}")
        return True
    except RuntimeError as e:
        print(f"❌ FlashAttention-2 사용 불가: {e}")
        print("   → model/attention.py 에서 sdpa_kernel 컨텍스트로 강제 wrap 필요.")
        return False


if __name__ == "__main__":
    ok = verify()
    raise SystemExit(0 if ok else 1)

# MoE Transformer — 의존성 관리 가이드

> **목표:** 로컬(M2 Pro)과 Colab(A100)에서 의존성 충돌 없이 동일한 코드를 실행

---

## 1. 핵심 원칙

```
로컬 (M2 Pro)          Colab (A100)
    │                       │
    │ CPU 모드로            │ A100 + BF16
    │ 코드 로직만 검증      │ 전체 학습
    │                       │
    의존성 독립 관리        Colab 기본 탑재 활용
    (venv)                  + pip install 최소화
```

**두 환경은 별개의 venv/런타임에서 운영합니다.**  
로컬에서 설치한 패키지가 Colab에 영향을 주지 않고, 그 반대도 마찬가지.

---

## 2. 로컬 (M2 Pro) — uv 가상환경

### 설치

```bash
cd ~/Desktop/moe-transformer-local
uv pip install -r requirements-local.txt
```

### 검증

```bash
.venv/bin/python poc_smoke_test.py
# 출력: 🎯 ALL DEPENDENCY CHECKS PASSED
```

### 로컬 의존성 목록 (`requirements-local.txt`)

| 패키지 | 최소 버전 | 용도 |
|:-------|:--------:|:-----|
| torch | 2.0.0 | 텐서 연산, CPU/MPS 모드 |
| accelerate | 0.28.0 | mixed precision 관리 (로컬은 mixed_precision="no") |
| transformers | 4.40.0 | PreTrainedTokenizerFast 래핑 |
| datasets | 2.18.0 | FineWeb-edu 로드 |
| tokenizers | 0.19.0 | BPE 직접 학습 |
| numpy | — | 통계/분석 |
| matplotlib | — | Loss 곡선 그래프 |
| tqdm | — | 진행바 |

### 주의: transformers 5.x

PoC 기준 transformers 5.9.0까지 정상 동작 확인.  
단, 5.x부터 `PreTrainedTokenizerFast` 생성 시 **`unk_token`이 반드시 vocab에 존재**해야 encode 가능.  
→ BPE 학습 시 `special_tokens`에 `<unk>`를 포함하면 자동 해결.

---

## 3. Colab (A100) — pip install 최소화

### Colab 기본 탑재 (설치 불필요)

```
Python 3.10
torch (CUDA, 최신 버전)
numpy
matplotlib
tqdm
```

### Colab Cell 0에서 설치 (순서가 생명!)

```python
# ★ 1. torch를 CUDA 버전으로 고정 (CPU-only 덮어쓰기 방지)
!pip install torch --index-url https://download.pytorch.org/whl/cu124 -q --upgrade

# ★ 2. 나머지 (torch는 이미 고정되어 건드리지 않음)
!pip install accelerate datasets tokenizers wandb -q

# ★ 3. 검증
import torch
assert torch.cuda.is_available(), "CUDA torch 설치 실패!"
```

> **⚠️ 절대 `!pip install torch`만 실행하지 말 것** — CPU-only torch로 덮어씀  
> **⚠️ 절대 accelerate를 torch보다 먼저 설치하지 말 것** — accelerate가 CPU-only torch를 끌어옴

| 패키지 | 설치 필요? | 이유 |
|:-------|:---------:|:-----|
| accelerate | ✅ | 미탑재. BF16 + 단일 GPU 관리 |
| datasets | ✅ | 미탑재 (Colab에 없을 수 있음) |
| tokenizers | ✅ | 미탑재. BPE 학습용 |
| wandb | ✅ (선택) | Loss 원격 로깅 |

### Colab 의존성 목록 (`requirements-colab.txt`)

```
accelerate>=0.28.0
datasets>=2.18.0
tokenizers>=0.19.0
wandb
```

---

## 4. PoC 검증 결과 (2026-06-01)

### 로컬 (M2 Pro, macOS 15.6.1, arm64)

| 패키지 | 설치 버전 | PoC 결과 |
|:-------|:--------:|:--------:|
| Python | 3.11.15 | ✅ |
| torch | 2.12.0 | ✅ matmul, RMSNorm |
| accelerate | 1.13.0 | ✅ device=mps, prepare OK |
| transformers | 5.9.0 | ✅ BPE → HF 변환 → encode/decode |
| datasets | 4.8.5 | ✅ Dataset.from_dict |
| tokenizers | 0.22.2 | ✅ BPE 학습, vocab=32K |

6/6 ALL PASSED.

### Colab (A100, 예상)

Colab의 Python 3.10 + CUDA torch 환경에서 추가 검증 필요.  
단, 사용하는 API(torch Tensor, Accelerator, PreTrainedTokenizerFast, Dataset)는 모두 Python 3.9~3.12에서 동일하게 동작.

---

## 5. 의존성 충돌 시나리오별 대응

| 상황 | 증상 | 해결 |
|:----|:----|:-----|
| `transformers` 버전 차이 | `PreTrainedTokenizerFast` 생성 실패 | `from_pretrained()`로 저장된 토크나이저를 로드하면 버전 영향 없음 |
| Colab GPU 드라이버 mismatch | `torch.cuda.is_available()=False` | Colab → 런타임 → 런타임 유형 변경 → A100 다시 선택 |
| Accelerate 버전 차이 | `mixed_precision` 옵션 인식 실패 | `accelerate config` 재설정 or `Accelerator()` 인자 생략 |
| Colab 기본 numpy 구버전 | datasets 로드 시 경고 | 기능상 문제 없음. 무시 |
| `uv pip install` 실패 | 네트워크 / index 오류 | `uv pip install --index-url https://pypi.org/simple/ ...` 로 fallback |

---

## 6. Colab ↔ 로컬 코드 동기화 주의사항

```python
# ❌ 절대 하지 말 것
if torch.backends.mps.is_available():
    device = "mps"           # 로컬에서만 동작
elif torch.cuda.is_available():
    device = "cuda"          # Colab에서만 동작

# ✅ 이렇게 할 것
from accelerate import Accelerator
accelerator = Accelerator()  # 자동 탐지 (로컬: mps/cpu, Colab: cuda)
```

로컬 디버깅과 Colab 학습 간 코드 차이는 **`Accelerator()`가 전부 처리**합니다.  
`device = "cuda"` 같은 하드코딩만 피하면 됩니다.

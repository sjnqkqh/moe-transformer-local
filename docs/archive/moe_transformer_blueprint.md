# MoE Transformer 청사진 (v1.0)

> **목적:** 트랜스포머와 MoE를 직접 구현하며 딥러닝 개념을 탑다운 방식으로 학습  
> **타겟:** 120M 파라미터급 MoE Transformer, Colab A100 기반 BF16 학습  
> **작성일:** 2026-06-01

---

## 1. 코어 아키텍처

**Transformer Decoder-only, 8층, 짝수층만 MoE 교체 (4 MoE + 4 Dense)**

| 항목 | 값 | 비고 |
|------|:--:|------|
| 층 구조 | 8층 (짝수 = MoE, 홀수 = Dense) | Interleaved 배치 |
| d_model | **768** | 임베딩 차원 |
| Attention Heads | **8개** | 96 dim/head |
| Context Window | **1024** | RoPE 기반 |
| Normalization | **Pre-RMSNorm** | LayerNorm 대비 경량, Pre 구조가 학습 안정성 ↑ |
| Positional Encoding | **RoPE** | 현대 트랜스포머 표준. 상대 위치 정보 내재 |
| FFN Activation | **SwiGLU** | GELU 대비 1~2% PPL 우위. d_ff 보정 필요 |
| d_ff (Dense FFN) | **2048** | d_model × 8/3 (SwiGLU는 Gate 2개이므로 보정) |
| 어휘 크기 | **32,000** | BPE 토크나이저에서 결정 |

### 추정 파라미터

| 구성요소 | 파라미터 |
|---------|:--------:|
| Token Embedding (untied) | 24.6M |
| 8층 Attention (QKV + Output) | 25.9M |
| 4층 Dense FFN (SwiGLU) | 18.9M |
| 4층 MoE FFN (4 experts × SwiGLU) | 75.5M |
| RMSNorm + 기타 | ~0.5M |
| **Total (emb tied)** | **~120M** |
| **Total (emb untied)** | **~145M** |

---

## 2. MoE 라우팅 명세

**핵심 원칙: MoE는 "더 많은 파라미터"가 아닌 "더 효율적인 파라미터 사용"이 목적**

| 항목 | 값 | 설명 |
|------|:--:|------|
| 라우팅 방식 | **Top-K (K=2)** | 각 토큰이 상위 2개 expert 선택 |
| Expert 수 | **4개** | MoE 레이어당 |
| Expert FFN | **SwiGLU, d_ff=2048** | Dense FFN과 동일 구조 |
| Capacity 제한 | **없음** | 모든 토큰 정상 라우팅, 토큰 드롭 X |
| Shared Expert | **없음** | 추후 확장 고려 (DeepSeek 스타일) |

### Auxiliary Loss — 2종 병용

Load Balancing Loss만으로는 라우터 logit 폭주를 막을 수 없어 Z-Loss를 함께 사용.

| Loss | 계수 | 목적 |
|------|:---:|------|
| **Load Balancing Loss** | α = 0.01 | Expert 간 사용률 균등화 |
| **Z-Loss** | β = 0.001 | Router logit 폭주 방지 (BF16 안정성) |

```
total_loss = main_loss + 0.01 × aux_loss + 0.001 × z_loss
```

---

## 3. 하드웨어 및 인프라

**로컬 = 디버깅 전용, 클라우드 = 실제 학습**

| 환경 | 하드웨어 | 용도 | 연산 정밀도 | 비고 |
|:----:|---------|------|:----------:|------|
| **로컬** | MacBook M2 Pro | 코드 로직 검증, 순전파/역전파 확인 | **FP32** (CPU 모드) | MPS BF16 미지원 + MoE 연산 일부 CPU fallback |
| **Colab** | A100 40GB | 전체 데이터셋 학습, HP 튜닝 | **BF16** | Accelerate + mixed_precision |

### 소프트웨어 스택

```
Python 3.10+
├── PyTorch 2.4+
├── Hugging Face Accelerate (단일 GPU, bf16)
├── tokenizers (BPE 학습용)
├── transformers (PreTrainedTokenizerFast 변환)
├── datasets (FineWeb-edu 로드)
└── wandb (로깅, 선택)
```

### 워크플로우

```
로컬 (CPU/FP32)           Colab (A100/BF16)
     │                         │
     │ PyTorch 코드 작성       │
     │ dummy batch (n=2~4)     │
     │ 로직만 검증             │
     │                         │
     └─────── 코드 복사 ──────→│
                               │ full 데이터 학습
                               │ HP 튜닝
                             ← Colab: 전체 학습 + 로깅
```

---

## 4. 결과물 저장 전략 + 로깅 체계

**Colab은 일시적 VM — 아무 조치 없으면 모든 파일이 사라짐**

| 저장소 | 용도 | 특징 |
|:------:|------|------|
| **Google Drive** | Colab 작업 공간 (체크포인트, 로그, 토크나이저) | 마운트 후 파일 읽기/쓰기, 런타임 교체에도 생존 |
| **로컬 MacBook Desktop** | 최종 결과물 (보고서, 그래프, 설계 문서) | Colab → `files.download()` 로 내려받기 |

### Google Drive 디렉토리 구조

```
/content/drive/MyDrive/moe_project/
├── checkpoints/
│   ├── moe_step500.pt
│   └── moe_final_step5000.pt
├── tokenizer/
│   ├── my_bpe_tokenizer.json
│   └── tokenizer_config.json
├── logs/
│   ├── metrics_r001_moe.jsonl ← 실험별 JSON Lines (100 step마다 1줄)
│   ├── metrics_r002_moe_aux001.jsonl     ← 실험 구분 = 파일 분리
│   ├── experiments_index.json            ← 전체 실험 목록 및 메타 정보
│   ├── events/                           ← 이벤트 로그 (체크포인트 저장, Collapse 등)
│   │   ├── r001_checkpoint_step1000.json
│   │   └── r002_expert_collapse.json
│   └── plots/                         ← 학습 완료 후 출력 그래프
│       ├── loss_curve.png
│       ├── expert_usage_over_time.png
│       └── router_entropy_over_time.png
├── data/
│   └── fineweb_edu_100k.arrow
└── reports/
    ├── evaluation_report.json         ← 자동화 평가 결과
    └── README.md
```

### 로깅 스펙 — 무엇을, 언제, 어디에 기록하는가

#### 학습 중 로깅 (100 step마다) → `logs/metrics.jsonl`에 append

| 필드 | 타입 | 출처 | 비고 |
|:----|:----:|:----|:-----|
| step | int | 학습 루프 카운터 | |
| main_loss | float | CrossEntropyLoss | |
| aux_loss | float | Load Balancing Loss | MoE 레이어만 |
| z_loss | float | Z-Loss | MoE 레이어만 |
| total_loss | float | main + α·aux + β·z | |
| ppl | float | exp(total_loss) | |
| lr | float | scheduler.get_last_lr() | |
| grad_norm | float | `torch.nn.utils.clip_grad_norm_()` | gradient 생존 확인 |
| expert_usage | [float, ...] | expert별 라우팅 비율(%) | 4개 값 리스트 |
| router_entropy | float | Expert 선택 분산도 | |
| gpu_memory_gb | float | `torch.cuda.max_memory_allocated()` | Colab A100 |
| tokens_per_sec | float | 직전 100 step 평균 | 학습 속도 추이 |
| epoch_progress | float | 현재 epoch 내 진행률 | |

#### 이벤트 로깅 (조건부) → `logs/events/`에 개별 JSON

| 이벤트 | 발생 조건 | 기록 내용 |
|:-------|:---------:|:----------|
| 체크포인트 저장 | 1000 step마다 | step, loss, expert_usage 스냅샷 |
| Expert Collapse 의심 | expert 사용률 < 5% | collapse된 expert ID, 발생 step |
| Loss Spike | step간 loss 증가율 > 20% | 이전 step loss, 현재 step loss, 직전 grad_norm |
| GPU OOM 위험 | memory > 35GB | 현재 배치 사이즈, seq_len, memory |

#### 저장 형식 선정 사유: JSON Lines + experiments_index.json

SQLite는 Google Drive 위에서 corruption 위험 (파일 잠금 충돌).  
현재 규모(실험 2~5회, 5,000 step = ~50KB)에서는 JSON Lines으로 충분.

| 요구사항 | JSON Lines + Index | SQLite |
|:---------|:------------------:|:------:|
| Colab + Drive 안전성 | ✅ append-only, 안전 | ❌ corruption 위험 |
| 여러 실험 조회 | ✅ index.json으로 | ✅ SQL |
| 사람이 읽기 | ✅ | ❌ |
| 구현 복잡도 | 낮음 | 중간 |

#### 학습 완료 후 평가 로깅 → `reports/evaluation_report.json`

학습이 끝난 후 evaluate.py가 생성하는 단일 JSON.  
「평가 기준」 문서의 5계층 평가 결과를 모두 포함.

---

### 체크포인트 주기

```
매 1000 step → Google Drive에 저장
→ 런타임 끊겨도 마지막 체크포인트부터 이어서 학습 가능

새 런타임 연결 시:
1. Drive 마운트
2. 가장 최근 체크포인트 탐색
3. 로드 후 이어서 학습
```

### Colab 초기화 코드 (매 런타임 첫 셀 — 이 셀 하나면 끝)

**⚠️ torch 설치 순서가 생명입니다. accelerate보다 먼저, CUDA index로 설치하세요.**

```python
# Cell 0: Setup
# ★ 1. torch를 CUDA 버전으로 고정
!pip install torch --index-url https://download.pytorch.org/whl/cu124 -q --upgrade

# ★ 2. 나머지 (torch는 이미 고정)
!pip install accelerate datasets tokenizers wandb -q

# ★ 3. 검증
import torch
assert torch.cuda.is_available(), "CUDA torch 설치 실패!"
print(f"✅ torch {torch.__version__}, GPU: {torch.cuda.get_device_name()}")

# ★ 4. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/moe_project"
CKPT_DIR = f"{PROJECT_DIR}/checkpoints"
LOG_DIR = f"{PROJECT_DIR}/logs"
TOKENIZER_DIR = f"{PROJECT_DIR}/tokenizer"
DATA_DIR = f"{PROJECT_DIR}/data"
for d in [CKPT_DIR, LOG_DIR, TOKENIZER_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
print("✅ Drive ready at", PROJECT_DIR)
```

이후 모든 학습 결과 — 체크포인트, 로그, 토크나이저, 전처리된 데이터 — 는 Google Drive의 `moe_project/` 아래 저장되어, 런타임이 끊겨도 생존하고 새 런타임에서 이어할 수 있다.

---

## 5. 데이터셋 (Google Drive에 저장)

| 용도 | 데이터셋 | 출처 | 크기 | Drive 저장 경로 |
|:----:|---------|------|:---:|:--------------:|
| 토크나이저 학습 | FineWeb-edu 50K 문서 | `HuggingFaceFW/fineweb-edu` | ~58MB | `$TOKENIZER_DIR/` |
| 모델 사전학습 | FineWeb-edu 100K 문서 | `HuggingFaceFW/fineweb-edu` | ~1~2GB | `$DATA_DIR/train.arrow` |

토크나이저 학습 후 결과물 → `$TOKENIZER_DIR/` (`/content/drive/MyDrive/moe_project/tokenizer/`)
모델 학습 데이터 전처리 후 → `$DATA_DIR/` (`/content/drive/MyDrive/moe_project/data/`)

---

## 6. 토크나이저 명세

```
알고리즘: BPE (Byte Pair Encoding)
vocab_size: 32,000
Special Tokens: <unk>, <s>, </s>, <pad>
Pre-tokenizer: ByteLevel (add_prefix_space=False)
Decoder: ByteLevel
```

**구현 흐름 (전체 Colab에서, 결과물은 Google Drive):**

```
① HuggingFace datasets로 FineWeb-edu 50K 문서 로드
② tokenizers.BPE() 생성
③ train_from_iterator() 실행
④ → my_bpe_tokenizer.json → $TOKENIZER_DIR/ 저장 (Google Drive)
⑤ PreTrainedTokenizerFast로 래핑 → $TOKENIZER_DIR/ 저장 (Google Drive)
⑥ 모델에서 f"{TOKENIZER_DIR}/..." 경로로 로드
```

---

## 7. 평가 계획 (전체 문서: `moe_evaluation.md`)

**이 모델이 120M이라 ChatGPT는 절대 안 나온다. 학습 목적에 충실한 판단 기준:**

### 판단 기준 (5계층)

| 계층 | 지표 | 최소 통과 기준 |
|:----:|:----|:--------------:|
| **Tier 1** 학습 건전성 | Loss 감소, Gradient Norm, PPL 단조성 | Step 5000 Loss < 초기 50% |
| **Tier 2** 언어 모델 품질 | Validation PPL, Token Accuracy | **PPL < 50** |
| **Tier 3** MoE 건강 | Load Balance CV, Expert Collapse, Router Entropy | **CV < 0.3**, Collapse 없음 |
| **Tier 4** 효율성 | 추론 속도, GPU 메모리 | 절대값 기록 (tokens/sec, GB) |
| **Tier 5** 확장 분석 | Expert 전문성 프로파일 (선택) | Expert 간 토큰 분화 |

### 자동화 평가 파이프라인

```bash
python train/evaluate.py \
    --checkpoint $CKPT_DIR/moe_final_step5000.pt \
    --output $LOG_DIR/evaluation_report.json
```

**한 줄 요약:** PPL 50 이하, Expert Collapse 없음, CV 0.3 이하 → "120M MoE가 정상 동작했다"

---

## 8. 구현 로드맵

---

## 8. 구현 로드맵

```
Phase 0: 인프라 셋업 + 토크나이저 (당일)
├── Colab + Google Drive 마운트 + Accelerate 설정
├── BPE 토크나이저 학습 → $TOKENIZER_DIR/ 저장
├── FineWeb-edu 전처리 → $DATA_DIR/ 저장
└── save/load/log 함수 준비

Phase 1: MoE Transformer 블록 구현 (2~3일)
├── Pre-RMSNorm + RoPE + SwiGLU + Attention
├── MoERouter (Top-2) + ExpertFFN (SwiGLU × 4)
├── Load Balancing Loss + Z-Loss
├── MoETransformerBlock (Attention + MoEFFN)
├── MoETransformer (8층, interleaved) 조립
└── 로컬 dummy data 역전파 검증

Phase 2: MoE 학습 (Colab A100) (2~3일)
├── 5000 step 학습 → PPL / Expert Usage / Router Entropy 기록
├── validation PPL 측정
└── evaluation_report.json 자동 생성

Phase 3: 확장/분석 (선택) (1~2일)
├── Expert 전문성 분석
├── 필요시 변주 실험 (aux 계수, top-1 vs top-2)
└── 최종 보고서 작성
```

---

## 9. 키워드 정리

| 용어 | 설명 |
|------|------|
| **MoE** | Mixture of Experts. 여러 FFN(Expert)을 두고 토큰마다 다른 Expert 활성화 |
| **Top-K** | 각 토큰이 K개의 Expert만 활성화하는 라우팅 방식 |
| **Auxiliary Loss** | Expert 사용률 균등화를 위해 메인 Loss에 추가하는 보조 Loss |
| **Load Balancing** | 모든 Expert가 비슷한 수의 토큰을 처리하도록 강제하는 기법 |
| **Z-Loss** | Router logit의 magnitude를 제한해 BF16 학습 안정화 |
| **Expert Collapse** | 특정 Expert만 계속 선택되어 나머지가 죽는 현상 |
| **SwiGLU** | Swish-gated Linear Unit. GELU보다 우수한 FFN 활성화 함수 |
| **RoPE** | Rotary Position Embedding. 상대 위치 정보를 attention score에 내재화 |
| **RMSNorm** | Root Mean Square Normalization. LayerNorm에서 mean 제거 → 경량화 |
| **Interleaved** | Dense 층과 MoE 층을 번갈아 배치하는 방식 |

---

*이 문서는 실제 구현이 진행됨에 따라 갱신됩니다.*

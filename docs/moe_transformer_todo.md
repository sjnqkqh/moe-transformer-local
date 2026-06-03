# MoE Transformer — 구현 TODO

> **목표:** Transformer Decoder-only + MoE를 직접 구현하여 MoE의 정상 동작과 Load Balancing 현상을 관찰  
> **모델:** 120M, 8층 (4 MoE + 4 Dense, interleaved), d_model=768, Top-2, SwiGLU, RoPE, Pre-RMSNorm  
> **하드웨어:** 로컬 M2 Pro (디버깅) + Colab A100 40GB (학습)

---

## Phase 0: 인프라 셋업 + 토크나이저

### □ 0.1 Colab 노트북 준비 + Google Drive 연결

- [ ] Colab에 새 노트북 생성 → `moe_transformer.ipynb`
- [ ] 런타임 → A100 변경 확인
- [ ] **Colab 의존성 설치 (Cell 0에서 1회) — 순서가 중요**
  ```python
  # ★ 순서 1: torch를 CUDA 버전으로 고정 (CPU-only로 덮어쓰기 방지)
  !pip install torch --index-url https://download.pytorch.org/whl/cu124 -q --upgrade
  
  # ★ 순서 2: 나머지 설치 (torch는 이미 고정되었으므로 건드리지 않음)
  !pip install accelerate datasets tokenizers wandb -q
  
  # 검증
  import torch
  assert torch.cuda.is_available(), "CUDA torch 설치 실패!"
  print(f"✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
  print(f"   GPU: {torch.cuda.get_device_name()}")
  print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
  ```
  > **⚠️ 절대 하지 말 것:** `!pip install torch`만 단독 실행 (CPU-only 버전 받아옴)  
  > **⚠️ 절대 하지 말 것:** accelerate/tokenizers 설치를 torch보다 먼저 실행 (의존성 충돌)
- [ ] Google Drive 마운트
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- [ ] 프로젝트 디렉토리 생성
  ```python
  import os
  PROJECT_DIR = "/content/drive/MyDrive/moe_project"
  CKPT_DIR = f"{PROJECT_DIR}/checkpoints"
  LOG_DIR = f"{PROJECT_DIR}/logs"
  TOKENIZER_DIR=f"{PR..."
  DATA_DIR = f"{PROJECT_DIR}/data"
  for d in [CKPT_DIR, LOG_DIR, TOKENIZER_DIR, DATA_DIR]:
      os.makedirs(d, exist_ok=True)
  ```
- [ ] accelerate 설정 (Colab 셀에서)
  ```python
  from accelerate import Accelerator
  accelerator = Accelerator(mixed_precision="bf16")
  ```
- [ ] (선택) wandb 로그인
  ```bash
  !wandb login
  ```

### □ 0.2 저장 체계 + 로깅 + 이어하기 로직

- [ ] 전역 경로 상수 (0.1에서 이미 정의했으면 그대로 사용)
- [ ] 체크포인트 저장 함수 작성 (Google Drive에 저장)
  ```python
  def save_checkpoint(model, optimizer, scheduler, step, loss, name):
      path = f"{CKPT_DIR}/{name}_step{step}.pt"
      torch.save({
          'step': step, 'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'loss': loss,
      }, path)
  ```
- [ ] 체크포인트 불러오기 함수 작성 (Google Drive에서 탐색)
  ```python
  def load_latest_checkpoint(model, optimizer, scheduler, pattern):
      ckpts = glob.glob(f"{CKPT_DIR}/{pattern}*.pt")
      if not ckpts: return 0
      latest = max(ckpts, key=os.path.getmtime)
      ckpt = torch.load(latest)
      model.load_state_dict(ckpt['model_state_dict'])
      optimizer.load_state_dict(ckpt['optimizer_state_dict'])
      scheduler.load_state_dict(ckpt['scheduler_state_dict'])
      return ckpt['step']
  ```
- [ ] log_metrics(step, metrics_dict) — JSON Lines → metrics_{run_id}_{name}.jsonl
- [ ] log_event(event_name, event_data) — 조건부 이벤트 → logs/events/
- [ ] collect_metrics(model, ...) — 13개 메트릭 dict 수집 (main_loss, aux_loss, z_loss, total_loss, ppl, lr, grad_norm, expert_usage, router_entropy, gpu_memory_gb, tokens_per_sec)
- [ ] init_experiment(run_id, name, config) / complete_experiment(run_id, final_metrics) / list_experiments()
- [ ] Colab 최초 셋업 셀 완성 (매 런타임마다 이 셀 하나만 실행)
  ```python
  # Cell 0: Setup
  !pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
  !pip install accelerate datasets tokenizers wandb -q
  from google.colab import drive
  drive.mount('/content/drive')
  PROJECT_DIR = "/content/drive/MyDrive/moe_project"
  ...  # 디렉토리 생성 + 검증
  print("✅ Drive ready at", PROJECT_DIR)
  ```

### □ 0.3 BPE 토크나이저 학습 (→ Google Drive에 저장)

- [ ] FineWeb-edu 데이터 로드 (50K 문서)
- [ ] BPE 토크나이저 생성 및 학습 (vocab=32000, special tokens 포함)
- [ ] HuggingFace 포맷 변환 → `$TOKENIZER_DIR/` 저장
- [ ] 토크나이저 encode/decode 테스트

### □ 0.4 학습 데이터 준비 (→ Google Drive에 저장)

- [ ] FineWeb-edu 100K 인코딩 → block_size=1024 청크 → `$DATA_DIR/` 저장
- [ ] Train/Val 분할 (90:10) → DataLoader 연결
- [ ] ~92M 토큰, A100 BF16 약 2~3시간 예상

---

## Phase 1: MoE Transformer 블록 구현

### □ 1.1 RMSNorm 구현

**파일 생성:** `model/normalization.py`

- [ ] `class RMSNorm(nn.Module)` — Pre-RMSNorm. LayerNorm에서 mean 제거한 경량 버전

### □ 1.2 RoPE 구현

**파일 생성:** `model/rope.py`

- [ ] `precompute_freqs_cis(dim, max_seq_len, theta=10000.0)` — sin/cos 테이블 생성
- [ ] `apply_rotary_emb(x, freqs_cis)` — Q, K에 회전 변환 적용

### □ 1.3 SwiGLU FFN 구현

**파일 생성:** `model/ffn.py`

- [ ] `class SwiGLU(nn.Module)` — gate=SiLU(x@W1), value=x@W2, output=(gate*value)@W3
- [ ] `class DenseFFN(nn.Module)` — SwiGLU wrapper. 홀수층용

### □ 1.4 MultiHeadAttention 구현

**파일 생성:** `model/attention.py`

- [ ] `class MultiHeadAttention(nn.Module)` — QKV projection → head split → RoPE → scaled dot-product → output

### □ 1.5 Dense TransformerBlock 구현

**파일 생성:** `model/transformer_block.py`

- [ ] `class TransformerBlock(nn.Module)` — 홀수층용. Attention + DenseFFN + Pre-RMSNorm ×2

### □ 1.6 MoE Router 구현

**파일 생성:** `model/moe_router.py`

- [ ] `class MoERouter(nn.Module)` — Top-2 gating
- [ ] `load_balancing_loss(router_logits, ...)` — Expert 사용률 균등 패널티 (α=0.01)
- [ ] `z_loss(router_logits)` — Router logit 안정화 (β=0.001)

### □ 1.7 Expert FFN × 4

**파일 생성:** `model/moe_ffn.py`

- [ ] `class ExpertFFN(nn.Module)` — DenseFFN과 동일한 SwiGLU 구조 × 4
- [ ] `class MoEFFN(nn.Module)` — Router + 4 Expert. Capacity 제한 없이 분배

### □ 1.8 MoE TransformerBlock 구현

**파일 생성:** `model/moe_layer.py`

- [ ] `class MoETransformerBlock(nn.Module)` — 짝수층용. Attention + MoEFFN + Pre-RMSNorm ×2
- [ ] Forward: output, aux_loss, z_loss 반환

### □ 1.9 MoE Transformer (8층, interleaved) 조립

**파일 생성:** `model/moe_transformer.py`

- [ ] `class MoETransformer(nn.Module)` — 짝수층=MoE, 홀수층=Dense. 8층.
- [ ] Forward: logits, total_aux_loss, total_z_loss 반환

### □ 1.10 로컬 디버깅 (M2 Pro, CPU 모드)

**선행 조건:** `cd ~/Desktop/moe-transformer-local && uv pip install -r requirements-local.txt`

- [ ] dummy batch (2, 8) forward 검증 — shape (2, 8, 32000)
- [ ] 역전파 모든 파라미터 gradient 존재 확인
- [ ] 파라미터 수 ~120M 확인
- [ ] MoE 레이어 expert 사용률 출력 (4개 expert 전부 선택되는지)

### □ 1.11 Phase 1 검증 (Pass 기준)

- [ ] 모델 ~120M 파라미터
- [ ] 모든 gradient 존재
- [ ] Aux Loss > 0, Z-Loss > 0
- [ ] Expert Collapse 없음

---

## Phase 2: MoE 학습 (Colab A100)

### □ 2.1 MoE 전용 로깅 세팅

- [ ] `RoutingHook` — 각 expert 사용률 실시간 캡처
- [ ] `collect_metrics()`에 expert_usage + router_entropy 통합
- [ ] Expert Collapse 탐지 → `log_event("expert_collapse", ...)`

### □ 2.2 Colab 학습 실행

- [ ] 실험 등록 (`init_experiment`)
- [ ] 이어하기 로직 포함 학습 루프 (5000 step, BF16)
- [ ] 100 step마다 metrics 기록
- [ ] 1000 step마다 체크포인트 + 이벤트 로그
- [ ] 완료 시 `complete_experiment()`

### □ 2.3 실험 결과 수집

- [ ] Loss / PPL 곡선 → `loss_curve.png`
- [ ] Expert Load Balancing (5 point) → `expert_usage_over_time.png`
- [ ] Router Entropy → `router_entropy_over_time.png`
- [ ] 추론 속도 benchmark (tokens/sec)
- [ ] 자동 평가 스크립트 실행 → `evaluation_report.json`

### □ 2.4 Phase 2 검증 (Pass 기준, 전체 기준: `moe_evaluation.md`)

- [ ] **Tier 1**: Loss 감소 (Step 5000 Loss < Step 0 50%)
- [ ] **Tier 1**: Gradient Norm 안정
- [ ] **Tier 2**: Validation PPL < 50
- [ ] **Tier 3**: Expert Collapse 없음 (전체 expert 사용률 > 5%)
- [ ] **Tier 3**: Load Balance CV < 0.3
- [ ] **Tier 4**: 추론 속도 측정 완료
- [ ] `evaluation_report.json` 저장 완료

---

## Phase 3: 확장 + 분석 (선택)

### □ 3.1 Expert 전문성 프로파일링

- [ ] 각 Expert에 라우팅된 토큰 빈출 분석 → Expert별 Top-20 토큰 출력

### □ 3.2 변주 실험 (필요시)

- [ ] aux 계수 변경 (0.001 / 0.01 / 0.1) → `experiments_index.json`으로 비교
- [ ] (관심 있으면) Top-1 vs Top-2

### □ 3.3 최종 보고서 작성

- [ ] `REPORT.md` → `~/Desktop/` 저장. 개요, 아키텍처, 실험 설계, 결과, 교훈

---

## 체크포인트 저장 구조 (Google Drive 기준)

```
/content/drive/MyDrive/moe_project/
├── checkpoints/
│   ├── moe_step500.pt
│   └── moe_final_step5000.pt
├── tokenizer/
├── logs/
│   ├── metrics_r001_moe_aux001.jsonl
│   ├── experiments_index.json
│   ├── events/
│   │   └── r001_checkpoint_step.json
│   └── plots/
│       ├── loss_curve.png
│       ├── expert_usage_over_time.png
│       └── router_entropy_over_time.png
├── data/
└── reports/
    └── evaluation_report.json
```

## 예상 시간표

| Phase | 내용 | 예상 시간 | 비고 |
|:-----:|------|:--------:|------|
| 0 | 인프라 + 토크나이저 | 1~2시간 | Colab 셋업 + BPE 학습 |
| 1 | MoE 블록 구현 | 6~10시간 | 코딩 + 로컬 디버깅 |
| 2 | MoE 학습 | 4~6시간 | Colab 학습 + 분석 |
| 3 | 분석 + 보고서 | 2~4시간 | 전문성 분석, 시각화, 작성 |

**총 예상: 13~22시간** (학습 대기 시간 포함)

---

## 참고: 실험실 노트 템플릿

```
실험명: [moe_aux001 / moe_aux01 / ...]
날짜: 2026-06-XX | 시드: 42 | Step: 5000 | Batch: 32 | LR: 3e-4
aux_lambda: 0.01 | z_lambda: 0.001
Final PPL: XX.XX | Expert Usage: [xx%, xx%, xx%, xx%]
Load Balance CV: X.XXX | Collapsed Experts: []
Router Entropy: X.XXX | 추론 속도: XX,XXX tok/s
```

---

*이 문서는 구현 진행에 따라 체크박스를 업데이트하세요.*

# Dense Transformer

한국어 사전학습 및 대화형 응답을 위한 **약 1.3B 파라미터 규모의 Decoder-only Dense Transformer**와 학습/평가/배포 파이프라인 구현체입니다. 로컬 환경(MPS/CPU)에서 모델 로직과 데이터 파이프라인을 신속히 검증한 뒤, Google Colab A100 80GB(BF16) 환경에서 본 학습을 수행하도록 설계되었습니다.

---

## 1. 핵심 아키텍처 사양

- **모델 크기:** 약 1.36B 파라미터 (Untied Embedding 구조, 임베딩/LM Head 가중치 미공유)
- **레이어 구성:** 24개 Dense Transformer 블록
- **차원 사양:** $d_{model} = 2048$, Attention Head 16개 ($d_{head} = 128$), FFN 중간 은닉 차원 $d_{ff} = 5632$
- **컨텍스트 윈도우:** 최대 시퀀스 길이 2048
- **어텐션:** SDPA(`scaled_dot_product_attention`) 기반 — CUDA + BF16/FP16 환경에서 FlashAttention-2 커널 자동 활성화
- **위치 인코딩:** RoPE (Rotary Position Embeddings)
- **활성화 함수:** SwiGLU
- **정규화:** Pre-RMSNorm + 최종 RMSNorm
- **사전학습 dropout:** 0.0 (SFT 단계에서 0.05 권장)

---

## 2. 프로젝트 구조

```
├── model/                          # 모델 아키텍처 패키지
│   ├── config.py                   # DenseTransformerConfig (1.3B v8 기본값)
│   ├── dense_transformer.py        # 모델 총조립, gradient checkpointing 훅
│   ├── transformer_block.py        # Pre-RMSNorm 블록
│   ├── attention.py                # SDPA 기반 인과적 멀티헤드 어텐션
│   ├── ffn.py                      # SwiGLU FFN
│   ├── rope.py                     # RoPE 주파수 연산 및 적용
│   └── normalization.py            # RMSNorm
├── tokenizer/
│   └── train_tokenizer.py          # 한국어 BPE 토크나이저 (HF dataset:config 지원)
├── train/                          # 학습·평가·배포 파이프라인
│   ├── train.py                    # Accelerate 기반 학습 (grad_accum / compile / Early Stopping)
│   ├── evaluate.py                 # 체크포인트 perplexity 평가
│   ├── prepare_chat_data.py        # 한국어 다중 데이터셋 전처리 (chunk 분리·재개 가능)
│   ├── prepare_data.py             # 영어 FineWeb-edu 전처리 (구버전)
│   ├── verify_data_pipeline.py     # 데이터셋 sanity test (Cell 4 전 실행 권장)
│   ├── verify_flash_attention.py   # FlashAttention-2 백엔드 가능 여부 검증
│   ├── generate.py                 # 추론(텍스트 생성)
│   ├── greeting_finetune.py        # 인사말 SFT
│   ├── export_for_local.py         # Colab 체크포인트 → 로컬 추론용 변환
│   ├── local_debug.py              # 로컬 셰이프/그래디언트 검증
│   └── utils.py                    # 체크포인트 I/O, JSONL 메트릭 로깅, NumpyDataset
├── tests/                          # 단위 테스트
├── docs/                           # 설계·운영 문서
├── Transformer_Model.ipynb         # Colab 학습 노트북 (Cell 0~8)
└── requirements-local.txt          # 로컬 환경 의존성
```

---

## 3. 현재 개발 상태 (v8)

- **모델 v8 스케일업:** 162M → 1.3B (`d=2048 / 24L / 16H / d_ff=5632 / seq=2048`)
- **데이터 파이프라인 v8.2:** 500M 토큰 단위 chunk 분리 + fsync + 진행상태 추적 → Colab 세션 중단 후 자동 재개 (`prepare_chat_data.py`)
- **데이터셋:** `fineweb-2:kor_Hang`, `wikipedia-korean`, `korean_textbooks`, `namuwiki` 4종 혼합 — 한국어 vocab 커버리지 확보
- **학습 최적화 옵션:**
  - `--compile`: `torch.compile` 활성화 (A100 BF16에서 +25~40% 처리량)
  - `--grad_checkpoint`: 활성화 메모리 ~4배 절감 (속도 -25%)
  - AdamW `fused=True` 자동 활성화 (CUDA 가용 시)
  - SDPA 자동 FlashAttention-2 (외부 `flash-attn` 패키지 불필요)
- **학습 안정성:** Gradient accumulation, Cosine LR + min_lr 하한, Early Stopping(patience/min_delta), best checkpoint 보존, loss spike 이벤트 로깅
- **실행 관리:** KST 타임스탬프 폴더 자동 생성으로 신규 학습/재개 분리, `metrics_{run_id}.jsonl` 학습 메트릭 누적

---

## 4. 실행 및 테스트 방법

### 1) 가상환경 및 패키지 설치
```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements-local.txt   # 또는: pip install -r requirements-local.txt
```

### 2) 단위 테스트
```bash
python -m unittest discover -s tests
```

### 3) 로컬 셰이프/그래디언트 검증 (데이터 불필요)
```bash
python train/local_debug.py
```

### 4) 로컬 스모크 테스트 (E2E)
```bash
# 토크나이저
python tokenizer/train_tokenizer.py --smoke_test --output_dir tokenizer/test_output

# 데이터 전처리
python train/prepare_data.py --smoke_test \
    --tokenizer_dir tokenizer/test_output \
    --output_dir train/test_data --block_size 16

# 미니 학습 10스텝
python -m train.train --smoke_test \
    --run_id test_dense --name test_dense \
    --data_dir train/test_data --tokenizer_dir tokenizer/test_output \
    --project_dir test_project --block_size 16 --max_steps 10

# 평가
python -m train.evaluate --smoke_test \
    --ckpt_dir test_project/checkpoints --checkpoint_pattern dense_test_dense \
    --data_dir train/test_data --block_size 16 \
    --output_file test_project/reports/evaluation_report.json
```

### 5) Colab 본 학습 (A100 80GB)
`Transformer_Model.ipynb` 셀 순서로 실행:

| 셀 | 역할 |
|---|---|
| Cell 0~2 | 소스코드 업로드 → Drive에 복사 |
| Cell 3 | 한국어 BPE 토크나이저 학습 |
| **Cell 3-B** | 데이터 파이프라인 sanity test (Cell 4 전 검증 권장) |
| Cell 4 | 7B 토큰 전처리 — chunk 분리·재개 가능 |
| **Cell 5** | 1.3B 사전학습 (`--compile`, BF16, batch 8 × grad_accum 8) |
| Cell 5-B | KST 폴더 기준 학습 재개 |
| Cell 6 | (선택) 인사말 fine-tuning |
| Cell 7~8 | 추론 테스트, 로컬 배포용 체크포인트 변환 |

### 6) FlashAttention-2 가용성 사전 점검 (선택)
```bash
python train/verify_flash_attention.py
```
A100 BF16 환경에서 `✅ FlashAttention-2 사용 가능`이면 SDPA 자동 선택으로 충분합니다.

---

## 5. 학습 산출물

- `{project_dir}/{KST}/checkpoints/dense_{run_id}_step{N}.pt` — 주기 체크포인트
- `{project_dir}/{KST}/checkpoints/dense_{run_id}_best.pt` — best validation 체크포인트
- `{project_dir}/{KST}/logs/metrics_{run_id}.jsonl` — step별 메트릭 누적
- `{project_dir}/{KST}/logs/events/` — loss spike 등 이벤트 JSON

체크포인트는 모델·옵티마이저·스케줄러 상태를 함께 저장하며, `load_latest_checkpoint`가 패턴 매칭 + 수정 시각 기준으로 가장 최근 파일을 자동 복원합니다.

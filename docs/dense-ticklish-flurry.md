# Dense Transformer 전환 계획

## Context

현재 프로젝트는 162M 규모의 **MoE(Mixture of Experts) Transformer**로, 8개 레이어를 짝수=MoE / 홀수=Dense로 교차 배치한 구조다. MoE는 학습 안정화를 위한 보조 손실(load balancing + z-loss), 라우터, 다중 전문가 FFN, 라우팅 프로파일러 등 부가 메커니즘이 다수 필요해 코드 흐름이 복잡하고, forward 반환값이 5-tuple로 부풀어 있다.

**Dense로 전환하면 다음이 사라진다**:
- `MoERouter`, `MoEFFN`, `ExpertFFN`, `MoETransformerBlock` 클래스
- 보조 손실 2종(`load_balancing_loss`, `router_z_loss`)과 계수 곱셈
- `RoutingProfiler` 훅 시스템, 전문가 사용률/엔트로피/CV 계산
- `forward()` 반환값이 `(logits, loss, main_loss)` 3-tuple로 단순화
- `loss = main_loss` 단일항 (보조 손실 가중합 제거)

결과적으로 **모델 코드 ~250줄, 학습 코드 ~80줄, 평가 코드 ~50줄 감소** 예상. 학습 안정성은 라우터 붕괴 문제가 사라져 오히려 단순해지고, 추론 시 토큰당 활성화 파라미터가 늘어 표현력이 더 일관된다. 트레이드오프는 동일 활성 파라미터당 표현력은 MoE보다 낮다는 점.

사용자 요구는 **162M 파라미터 규모를 유지**하면서 전면 리네이밍(MoE→Dense)으로 완전 전환하고, MoE 브랜치는 별도 분기로 보관한다.

---

## 162M 파라미터 보정 계산

현재 MoE 162M 내역:
| 구성 | 파라미터 |
|------|---------|
| Token Embedding (32k×768) | 24.6M |
| LM Head untied (768×32k) | 24.6M |
| Attention 8 layers (4·d²×8) | 18.9M |
| Dense FFN 4 layers (SwiGLU, d_ff=2048) | 18.9M |
| MoE FFN 4 layers (4 experts × SwiGLU) | 75.5M |
| **합계** | **~162M** |

단순히 8개 모두 Dense(d_ff=2048)로 바꾸면 → **105.8M** (56M 부족).

**보정 옵션 → 권장: Option B (관행적 형태)**

| Option | n_layers | d_ff | 합계 | 비고 |
|--------|---------|------|------|------|
| A: d_ff만 확장 | 8 | **5120** | 162.4M | 최소 변경, 비정상적으로 넓은 FFN |
| **B: 관행적 확장** | **12** | **3072** | **162.4M** | GPT-2 small 계열 표준 비율 (d_ff=4·d_model) |
| C: 레이어만 확장 | 16 | 2048 | 162.4M | 깊은 모델, 학습 메모리 ↑ |

→ **Option B 권장**: d_ff=4·d_model이 트랜스포머 표준 비율이며, n_layers=12는 GPT-2 small과 동일해 비교 기준이 명확하다.

---

## 작업 절차

### Step 1: 브랜치 분기

```bash
git checkout -b moe-archive            # 현 MoE 상태 보존
git push -u origin moe-archive          # 원격에 백업
git checkout main
git checkout -b dense-conversion        # 작업 브랜치
```

이후 모든 변경은 `dense-conversion`에서 진행. main으로 머지 시점에 README/CLAUDE.md도 함께 갱신.

### Step 2: MoE 전용 파일 삭제 (3개)

- `model/moe_router.py`
- `model/moe_ffn.py`
- `model/moe_layer.py`

### Step 3: 모델 파일 리네이밍 및 단순화

- `model/moe_transformer.py` → `model/dense_transformer.py`
  - 클래스명: `MoETransformer` → `DenseTransformer`
  - 교차 배치 로직 제거 → 모든 레이어를 `TransformerBlock` 단일 타입으로 생성
  - `forward()` 반환값: `(logits, loss, main_loss, aux_loss, z_loss)` → `(logits, loss, main_loss)`
  - aux/z 손실 누적 코드 제거
  - 최종 loss 공식: `loss = main_loss` (계수 가중합 제거)
  - 파라미터 카운트: `n_layers=12`, `d_ff=3072`로 162M 보정

### Step 4: Config 스키마 변경

`model/config.py`:
- 클래스명: `MoETransformerConfig` → `DenseTransformerConfig`
- 필드 제거: `num_experts`, `k`
- 기본값 변경: `n_layers=12`, `d_ff=3072`
- `dropout`, `eps`, `vocab_size`, `d_model=768`, `n_heads=8`, `max_seq_len=1024` 유지

### Step 5: 학습 파이프라인 정리

`train/train.py`:
- `MoETransformerBlock`, `MoETransformer`, `MoETransformerConfig` import → Dense 버전으로 교체
- `RoutingProfiler` 클래스 전체 삭제 (train.py:27-65)
- 훅 등록/해제 코드 삭제 (train.py:151-157, 302-303)
- forward 언패킹: `logits, loss, main_loss, aux_loss, z_loss = model(...)` → `logits, loss, main_loss = model(...)`
- `metrics` 딕셔너리에서 `aux_loss`, `z_loss`, `expert_usage`, `router_entropy` 키 삭제
- `expert_collapse` 이벤트 로깅 제거 (train.py:281-288)
- 모델 파라미터 출력부에서 Dense FFN/MoE FFN 분리 출력을 단일 FFN 합산으로 통일 (train.py:103-122)
- argparse 그대로 유지 (run_id, batch_size 등은 영향 없음)

`train/evaluate.py`:
- MoETransformerBlock import 제거, MoETransformer→DenseTransformer 교체
- 라우팅 분석 훅 코드 제거 (evaluate.py:88-96, 140-162)
- forward 언패킹 단순화 (evaluate.py:113)
- 보고서에서 CV/expert usage 필드 제거, PPL만 남김

`train/local_debug.py`:
- forward 언패킹 단순화 (local_debug.py:99)
- aux_loss/z_loss assert 검증 삭제 (local_debug.py:105-114)
- 라우팅 균형 검증 블록 전체 삭제 (local_debug.py:159-205)
- 파라미터 카운트 출력 단순화

### Step 6: 테스트 재작성

기존 `tests/test_routing.py` 삭제. 나머지는 Dense 맞춤 재작성:

- `tests/test_config.py`: `DenseTransformerConfig` 기본값 검증, `num_experts/k` 필드가 더 이상 없음을 확인, 모델 초기화 호환성
- `tests/test_dropout.py`: 어텐션 + `SwiGLU/DenseFFN` 드롭아웃 모듈 존재 및 train/eval 모드 차이 검증 (`MoEFFN` 검증 부분만 교체)
- `tests/test_dataset.py`: 변경 불필요 (MoE 비의존)
- **추가**: `tests/test_forward.py` 신설 — 새 3-tuple 반환값, 파라미터 수가 162M ± 5M 범위인지, 인과적 마스킹 동작 검증

### Step 7: 문서 동기화

- `CLAUDE.md`: 아키텍처 섹션을 Dense 12-layer 구조로 수정, MoE/라우팅 섹션 삭제
- `README.md`: 모델 사양 표 갱신, `n_layers=12 / d_ff=3072 / 162M`로, MoE 관련 설명 제거
- `docs/`: MoE 전용 문서(`moe_transformer_blueprint.md`, `moe_transformer_todo.md`, `moe_training_results.md`)는 `docs/archive/`로 이동

---

## 영향 받는 파일 요약

| 동작 | 파일 |
|------|------|
| **삭제** | `model/moe_router.py`, `model/moe_ffn.py`, `model/moe_layer.py`, `tests/test_routing.py` |
| **리네이밍 + 수정** | `model/moe_transformer.py` → `model/dense_transformer.py`, `model/config.py` (클래스명) |
| **수정** | `train/train.py`, `train/evaluate.py`, `train/local_debug.py`, `tests/test_config.py`, `tests/test_dropout.py`, `CLAUDE.md`, `README.md` |
| **신설** | `tests/test_forward.py` |
| **이동** | `docs/moe_*.md` → `docs/archive/` |
| **무변경** | `model/transformer_block.py`, `model/attention.py`, `model/ffn.py`, `model/normalization.py`, `model/rope.py`, `tokenizer/`, `train/prepare_data.py`, `train/utils.py`, `tests/test_dataset.py` |

---

## 검증 방법

작업 완료 후 다음 순서로 end-to-end 검증:

```bash
# 1. 단위 테스트 (Dense 재작성본)
python -m unittest discover -s tests

# 2. 로컬 디버깅 — 모델 초기화, 파라미터 카운트 (~162M ±5M), 그래디언트 흐름
python train/local_debug.py
# 기대 출력: Total Parameters ≈ 162M, 모든 레이어 grad 정상

# 3. E2E 스모크 테스트 (10 step)
python tokenizer/train_tokenizer.py --smoke_test --output_dir tokenizer/test_output
python train/prepare_data.py --smoke_test --tokenizer_dir tokenizer/test_output --output_dir train/test_data --block_size 16
python -m train.train --smoke_test --run_id test_dense --name test_dense --data_dir train/test_data --tokenizer_dir tokenizer/test_output --project_dir test_project --block_size 16 --max_steps 10
python -m train.evaluate --smoke_test --ckpt_dir test_project/checkpoints --checkpoint_pattern moe_test_dense --data_dir train/test_data --block_size 16 --output_file test_project/reports/eval_dense.json

# 4. 손실 단조 감소 확인 — metrics_test_dense.jsonl에서 main_loss 추세 확인
```

**합격 기준**:
- 모든 unittest 통과
- `local_debug.py`가 162M ±5M 범위 출력 및 NaN/Inf 없는 그래디언트 보고
- 학습 스모크 10 step이 OOM/예외 없이 완료, `main_loss` 감소 추세
- evaluate가 PPL 수치를 산출하고 CV/expert 필드가 보고서에 없음

**회귀 점검**:
- `MoE`, `aux_loss`, `z_loss`, `router`, `expert` 등 키워드가 `model/`, `train/`, `tests/`에 0건 남았는지 grep 확인
- Checkpoint 패턴(`moe_{run_id}`)도 `dense_{run_id}`로 변경하는 경우 `train.py:166`, `evaluate.py`에서 일관성 점검

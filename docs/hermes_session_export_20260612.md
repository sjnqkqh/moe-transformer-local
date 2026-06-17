# 162M Dense Transformer 한국어 챗봇 — 버전별 종합 리포트

> **프로젝트 기간:** 2026-06-01 ~ 2026-06-12
> **목표:** 직접 Transformer를 구현하고 한국어 챗봇으로 발전시키는 딥러닝 학습 프로젝트
> **로컬 경로:** `~/Desktop/moe-transformer-local/`
> **Colab Drive:** `/content/drive/MyDrive/korean_chat/`
> **서버:** FastAPI (localhost:8000, M2 Mac MPS)

---

## 목차

1. [MoE Transformer 초기 구현 (v1~v3)](#1-moe-transformer-초기-구현-v1v3)
2. [Dense 전환 및 아키텍처 확정](#2-dense-전환-및-아키텍처-확정)
3. [v4: 최초 한국어 학습](#3-v4-최초-한국어-학습)
4. [v5: 하이퍼파라미터 튜닝](#4-v5-하이퍼파라미터-튜닝)
5. [v6: 인사말 Fine-Tune](#5-v6-인사말-fine-tune)
6. [로컬 배포 및 추론 테스트](#6-로컬-배포-및-추론-테스트)
7. [v7: Validation + Early Stopping + Dropout](#7-v7-validation--early-stopping--dropout)
8. [데스크탑 정리 및 파일 구조](#8-데스크탑-정리-및-파일-구조)
9. [최종 비교 및 향후 방향](#9-최종-비교-및-향후-방향)
10. [Overfitting/Underfitting 딥다이브 문서](#10-overfittingunderfitting-딥다이브-문서)

---

## 1. MoE Transformer 초기 구현 (v1~v3)

### 개요

FineWeb-edu 영어 데이터를 사용한 162M MoE Transformer. MoE 구조의 정상 동작 검증이 목적.

### 사용자 질문
> "바텀업 수업 + 탑다운 구현 병행. 현재까지 결정한 MoE 사양을 검토하고 청사진을 그려줘."
> "Dense FFN이 뭔지 설명해줘."
> "작성해준 문서 기반으로 별도의 TODO 문서를 작성해줘. 각 Step이 상세할수록 좋아."
> "코드 작성은 늦을수록 좋아. 현재까지 결정된 사항을 MD 파일로 작성해줘. 되도록 두괄식으로."

### 아키텍처 (초기 MoE)

```
8층 Transformer (짝수층 MoE, 홀수층 Dense interleaved)
├── d_model=768, 8 Head MHA, Context=1024
├── Pre-RMSNorm + RoPE + SwiGLU
├── MoE: Top-2, 4 Experts, Load Balancing Loss (α=0.01) + Z-Loss (β=0.001)
└── ~162M params (untied embeddings)
```

### 학습 결과 (Colab A100, BF16, FineWeb-edu sample-10BT, 1M docs)

|   버전   |    Steps    |  Block   | Batch |      기법      | 최저 Loss |     최저 PPL     | 비고        |
|:------:|:-----------:|:--------:|:-----:|:------------:|:-------:|:--------------:|-----------|
| **v1** |   0→5000    |   512    |   4   |    Basic     |  ~5.0   |      155       | 문법 습득 시작  |
| **v2** | 5000→10000  |   512    |   4   | Grad Accum 4 |  ~4.2   |       65       | 어휘 다양성 증가 |
| **v3** | 10000→15000 | **1024** |   4   | Grad Accum 4 | **3.5** | **33.09** (최저) |

### 생성 결과 (v3, argmax decoding)

| 프롬프트                      | 출력                                                                                                 |
|---------------------------|----------------------------------------------------------------------------------------------------|
| "The future of AI is"     | `a very important aspect of the future of AI. The future of AI is a very important aspect...` (반복) |
| "Machine learning models" | `the first step is to create a model that is used to model the model...` (반복)                      |

### MoE 건강도

- **Expert Collapse:** 0건 ✅
- **Load Balance CV:** ~0.0 (완전 균등) ✅
- **Aux Loss:** 2.0/레이어 (이론적 최적값) ✅
- **결론:** Aux Loss가 의도대로 완벽히 작동했으나, 162M/4Experts에서는 Dense 대비 명확한 우위를 체감하기 어려움

### 생성 한계

- Argmax decoding의 반복 생성 문제 → temperature sampling 필요
- 162M 모델의 근본적인 언어 모델링 능력 한계 확인
- 한국어 전혀 학습되지 않음 (ByteLevel 영어 기반 tokenizer)

---

## 2. Dense 전환 및 아키텍처 확정

### 전환 결정

MoE → Dense 전환의 이유:

- 162M/4Experts에서는 MoE가 Dense 대비 의미 있는 우위 없음
- 한국어 챗봇이라는 실용적 목표로 방향 전환

### 최종 아키텍처

```python
DenseTransformerConfig(
    vocab_size=32000,  # 한국어 BPE
    d_model=768,
    n_layers=12,  # MoE 8층 → Dense 12층
    n_heads=8,
    d_ff=3072,
    max_seq_len=512,  # 한국어 대화용으로 축소
    dropout=0.15  # v5부터 적용
)
```

### 데이터셋 (7개 한국어 + AI Hub)

| 데이터셋                                | 규모                    | 포맷                          |
|-------------------------------------|-----------------------|-----------------------------|
| nlpai-lab/kullm-v2                  | 152K rows             | instruction/output          |
| beomi/KoAlpaca-v1.1a                | ~10K+                 | instruction/output          |
| beomi/KoAlpaca-RealQA               | ~10K+                 | instruction/output          |
| kyujinpy/KOR-OpenOrca-Platypus-v3   | ~10K+                 | instruction/output          |
| JaeJiMin/korean_chat_friendly       | ~10K+                 | short_question/short_answer |
| FreedomIntelligence/sharegpt-korean | ~10K+                 | conversations               |
| BAEM1N/nanochat_korean              | 202K rows, **30.1GB** | text (parser 의심)            |
| AI Hub TL_session                   | 다양                    | 세션 기반 대화                    |

### 데이터 포맷

```
<user>질문<sep><assistant>답변</s>
```

모든 데이터를 이 포맷으로 통일.

### 학생 데이터 (greeting fine-tune용)

1,200쌍 (40종 × 30배) — 인사말 / 반가움 / 안부 / 소개 / 감정표현 / 대화유도

---

## 3. v4: 최초 한국어 학습

### v4 사용자 질문
> "스크린샷으로 nanochat 데이터를 추가했고 W&B 로그도 포함했어. 데이터가 올바르게 로드되고 있는지, epoch는 적절한지 파악해줘."

### 설정

| 항목         | 값                     |
|------------|-----------------------|
| Epochs     | 1                     |
| Warmup     | 500 steps (전체의 20.5%) |
| Dropout    | 0.1                   |
| LR         | 3e-4 → Cosine Decay   |
| Batch size | 32                    |
| Block size | 512                   |
| 총 steps    | ~2,435                |

### W&B 분석 결과

| 항목        | 값                                       |
|-----------|-----------------------------------------|
| 초기 Loss   | ~7.0                                    |
| 최종 Loss   | **~4.2**                                |
| 최종 PPL    | ~66.7                                   |
| Grad norm | 말미 변동성 증가 (0.42→0.48)                   |
| LR        | 3e-4 → ~1.5e-4 (warmup 500 → cos decay) |

### 발견된 문제점

1. **1epoch = 40M tokens** → Chinchilla ratio 0.25x (최적의 1.25%)
2. **Warmup 20.5%** → 유효 학습 구간이 너무 짧음
3. **Loss 4.2에서 plateau** → underfitting or epoch 부족?
4. **대화 vs 긴 문서** — nanochat_korean 30GB 포맷 불일치 의심
5. **Bingsu/KoAlpaca_v1.1a** → 존재하지 않는 데이터셋

### 생성 결과 (temperature 0.8)

- "안녕" → "질문자님의 궁금증을..." (지식iN 패턴 고착)
- kullm-v2(152K rows)가 지배적인 영향

---

## 4. v5: 하이퍼파라미터 튜닝

### v5 사용자 질문
> "폴더 내의 파일들을 스캔해서 현재 상황을 파악하고, 추가된 데이터셋을 기준으로 목적에 부합하는 데이터셋이 올바르게 학습되고 있는지, epoch는 어느정도가 적절할지 파악해줘."
> "epoch 3, warmup 200, dropout 0.15 적용해줘."

### 변경사항

| 항목             |     v4      |      v5      | 이유              |
|----------------|:-----------:|:------------:|-----------------|
| Epochs         |      1      |    **3**     | Underfitting 해결 |
| Warmup         | 500 (20.5%) | **200 (8%)** | 유효 학습 구간 확보     |
| Dropout        |     0.1     |   **0.15**   | 3epoch 과적합 방지   |
| LR             |    3e-4     |  3e-4 (유지)   | -               |
| Bingsu → beomi |      ❌      |      ✅       | 존재 확인           |
| Batch size     |     32      |      32      | 유지              |

### 학습 결과 (22K steps까지 진행)

| 항목            | 값               |
|---------------|-----------------|
|| 총 Steps       | 22,000 (중단)     |
|| 최종 Train Loss | **2.20**        |
|| 최종 Train PPL  | **9.07**        |
| 초기 Loss       | ~7.0            |
| LR at stop    | ~1.0e-4         |
| Grad norm     | 0.45~0.48 (안정적) |
| Loss spike    | 4회 발생 후 정상 회복   |

### Loss 곡선 추이

### 교훈: plateau를 underfitting으로 오해한 케이스

- v4에서 Loss 4.2에서 plateau → "epoch 부족인가 과적합인가?"
- 실제로는 epoch 1에서 **underfitting** — 22K step까지 늘리자 loss 2.20까지 하락
- "Loss가 높다고 overfitting이 아니다. plateau의 기울기를 확인하라."

---

## 5. v6: 인사말 Fine-Tune

### v6 사용자 질문
> "멀티턴 기반 대화를 진행하려면 어떤 변경사항들이 들어갈까? (+ 지금 내 상황에서 가능할까?)"
> "하나 추가로 하고 싶은건 특정한 데이터 (개발자 이창신, 양준렬, 카카오 테크 부트캠프) 에 대한 추가적인 정보를 집어넣고 싶다면 가능할까?"

### 목적

"안녕" 입력 시 "질문자님의 궁금증을..." 지식iN 패턴 대신 자연스러운 인사말 출력

### 데이터

40종 × 30배 = **1,200쌍** (인사말/반가움/안부/소개/감정표현/대화유도)

### 설정

| 항목         | 값                        |
|------------|--------------------------|
| Base model | v5 checkpoint (step 22K) |
| Epochs     | 5                        |
| Batch size | 8                        |
| LR         | 5e-5                     |
| Dropout    | 0.0 (설정 오류, 나중에 발견)      |
| 최종 Loss    | **0.1367** (인사말 1,200쌍 한정)      |

### 결과

- "안녕" → "안녕하세요! 편하게 물어봐 주세요." ✅
- "안녕하세요" → "네, 안녕하세요! 무엇을 도와드릴까요?" ✅
- temperature 1.0 + top_k 30으로 4가지 이상의 다양한 응답 확인

### 문제점 (사후 발견)

- **dropout=0.0** → 과적합 위험 (1,200쌍/162M = 0.0007%)
- 5epoch 고정, Early Stopping 없음
- Validation 분할 없음 (전체 데이터 학습)
- Train loss만 0.1367까지 떨어졌으나 실제 생성 품질 개선은 미미

---

## 6. 로컬 배포 및 추론 테스트

### 서버 구조

```
serve/
├── app.py              # FastAPI (port 8000)
├── model/
│   └── korean_chat.pt  # v6 export checkpoint
└── ... (HTML/JS 프론트엔드 포함)
```

### 실행 명령어

```bash
cd ~/Desktop/moe-transformer-local
uvicorn serve.app:app --reload --host 0.0.0.0 --port 8000
```

### 추론 파라미터 (최종 설정)

| 파라미터               |  값   | 비고                |
|--------------------|:----:|-------------------|
| temperature        | 1.0  | 0.8에서 올림 (다양성 확보) |
| top_k              |  30  | 상위 30개 토큰에서 샘플링   |
| top_p              | 0.95 | 누적 확률 95%         |
| repetition_penalty | 1.5  | 반복 방지             |
| max_new_tokens     | 100  | -                 |

### 토크나이저 이슈

- **문제:** 서버가 293 vocab 토크나이저로 로드됨
- **원인:** 로컬 smoke test용 tokenizer만 존재
- **해결:** Colab에서 32K tokenizer.zip 다운로드 → 덮어쓰기

### 응답 속도 (M2 Mac)

> "응답 속도가 어마어마하게 빠르네" — 사용자 피드백

---

## 7. v7: Validation + Early Stopping + Dropout

### v7 사용자 질문
> "내가 진행한 프로젝트에 얼리스탑과 검증이 포함되지 않았다는 걸 발견했는데 확인해줄래?"
> "이번에 한국어 챗봇 5epoch 학습 끝났어. Train Loss 1.93, Val Loss 2.66, PPL 6.86."
> "이제 일반 상식을 가진 챗봇 모델을 만들고 싶은데 300GB 데이터는 너무 터무니 없나?"

### 배경

이전 버전들(v4~v6)에는 Validation과 Early Stopping이 전혀 없었음.

- v6 greeting fine-tune에서 dropout=0.0 + 5epoch = 전형적인 overfitting 케이스
- Overfitting/Underfitting 딥다이브 학습 후 적용 결정

### 변경사항

#### train.py

| 항목              |     v5/v6      |            v7            |
|-----------------|:--------------:|:------------------------:|
| Validation      |      ❌ 없음      | ✅ val.npy 로드 + val loop  |
| Early Stopping  |      ❌ 없음      | ✅ patience=5, delta=1e-3 |
| Best checkpoint | ❌ (step 간격 저장) |  ✅ val_loss 최저 시점 별도 저장  |
| Dropout 기본값     |      0.1       |         **0.15**         |
| epochs 설정       |       3        |     **5** (ES가 찾도록)      |

#### greeting_finetune.py

| 항목             |     v6     |          v7          |
|----------------|:----------:|:--------------------:|
| 데이터 분할         |   전체 학습    |  90:10 train/val 분할  |
| Validation     |     ❌      |    ✅ val loss 계산     |
| Early Stopping |     ❌      |     ✅ patience=2     |
| Dropout        |    0.0     |       **0.15**       |
| 저장             | 마지막 weight | best epoch weight 복원 |

#### Transformer_Model.ipynb

|      항목       |    v6    |                  v7                   |
|:-------------:|:--------:|:-------------------------------------:|
|    Cell 수     |   10개    |          **9개** (Cell 2 제거)           |
|    학습 명령어     | epochs 3 | epochs 5 + val_every 500 + patience 5 |
| 인사말 fine-tune |    v6    |              v7_greeting              |
|     체크포인트     |  v5.pt   |            **v7_best.pt**             |

### v7 최종 학습 결과 (55,430 steps, 5epoch)

| 지표                |          값           |
|:------------------|:--------------------:|
| **Train Loss**    |      **1.925**       |
| **Train PPL**     |      **6.856**       |
| **Val Loss**      |      **2.658**       |
| **Val PPL**       |      **14.268**      |
| **Train-Val Gap** |      **0.733**       |
| Epochs            |    5 (Epoch 0~4)     |
| Steps             |        55,430        |
| 최종 LR             | ~0 (cosine decay 완료) |

### Loss 곡선 추이


### Early Stopping 미발동 원인

- Train-Val Gap이 0.73으로 유지 → 과적합 inflection point 도달 전
- dropout 0.15의 정규화 효과 + 데이터 충분
- 5epoch이 과적합을 유발할 만큼 많지 않음
- 만약 10~15epoch까지 갔다면 val loss가 반등했을 가능성 높음

---

## 8. 데스크탑 정리 및 파일 구조

### 최종 Desktop 구조

```
Desktop/
├── moe-transformer-local/            ← 전체 프로젝트
├── hermes_session_export_20260611.md  ← 6/11 세션 export
├── hermes_session_export_20260612.md  ← 6/12 세션 export (본 문서)
├── overfitting_underfitting_deepdive.md  ← 딥다이브 문서 (v2 간결판)
│
├── _archive/
│   ├── MoE/          ← MoE Transformer 관련 문서 (7개)
│   ├── colab_scripts/  ← Colab 셀별 추출 스크립트 (4개)
│   ├── session_exports/  ← 이전 세션 export (1개)
│   └── etc/          ← 기타 파일 (2개)
│
├── Screenshots/      ← 스크린샷/데모 영상 (6개)
└── learning/         ← 학습 자료 (4개)
```

### 프로젝트 코드 구조

```
moe-transformer-local/
├── model/
│   ├── dense_transformer.py   ← 12층 Dense Transformer
│   ├── config.py              ← DenseTransformerConfig
│   └── ffn.py                 ← SwiGLU FFN
├── train/
│   ├── train.py               ← 학습 (v7: +ES+Validation)
│   ├── greeting_finetune.py   ← 인사말 fine-tune (v7: +ES+dropout)
│   ├── generate.py            ← 추론 함수
│   ├── prepare_chat_data.py   ← 데이터 전처리
│   ├── prepare_data.py        ← FineWeb-edu 전처리
│   ├── evaluate.py            ← 평가
│   ├── export_for_local.py    ← 서버 export
│   └── utils.py               ← 공유 유틸리티
├── tokenizer/
│   └── korean_output/         ← 32K BPE tokenizer
├── serve/
│   └── app.py                 ← FastAPI 서버
└── Transformer_Model.ipynb    ← v7 Colab Notebook
```

---

## 9. 최종 비교 및 향후 방향

### 버전별 핵심 지표 비교

|   버전   |   Loss   |   PPL    | Val Loss |  Val PPL  | 비고                             |
|:------:|:--------:|:--------:|:--------:|:---------:|--------------------------------|
| **v1** |   5.0    |   155    |    -     |     -     | 영어 MoE, 첫 학습                   |
| **v2** |   4.2    |    65    |    -     |     -     | Grad accum 적용                  |
| **v3** |   3.5    |    33    |    -     |     -     | Context 1024, 최고 PPL           |
| **v4** |   4.2    |   66.7   |    -     |     -     | 한국어 첫 학습, 1epoch               |
|| **v5** | **2.20** | **9.07** |    -     |     -     | 22K steps, HP 튜닝               |
| **v6** |  0.137   |    -     |    -     |     -     | 인사말 fine-tune (overfit)        |
| **v7** | **1.93** | **6.86** | **2.66** | **14.27** | **55K steps, +ES+Val+dropout** |

### v7 vs 기존 모델 대비 개선율

| 지표         |   v5    |    v7     |     개선      |
|:-----------|:-------:|:---------:|:-----------:|
|| Train Loss |  2.20   | **1.925** | ↓ **12.7%** |
| Train PPL  |  9.07   | **6.86**  | ↓ **24.4%** |
| 실제 생성      | 지식iN 패턴 | 다양한 인사말✅  |   체감 개선 큼   |

### 향후 방향 논의

| 주제          | 논의 내용                                                                                                     |
|-------------|-----------------------------------------------------------------------------------------------------------|
| **멀티턴 대화**  | generate.py + app.py만 수정하면 가능, 모델 재학습 불필요 (AI Hub 데이터가 이미 멀티턴 패턴 포함)                                      |
| **지식 주입**   | 컨텍스트 주입 방식 (generate.py 수정) 우선, 필요시 knowledge fine-tune                                                   |
| **모델 스케일업** | 162M → 350M or 1B: A100 40GB에서 batch_size=8, grad_accum=4, block_size=256, gradient_checkpointing 조합으로 가능 |
| **데이터 확장**  | 300GB 데이터는 162M 대비 오버스펙. 1B 모델 기준으로도 FineWeb-edu 10BT(40GB) + 한국어 데이터 20~30GB면 충분                         |
| **GPU 메모리** | 활성화 메모리(activations)가 40GB의 대부분 차지. batch_size 반으로 줄이고 grad_accum 늘리면 메모리 50% 이상 절감 가능                    |

---

## 10. Overfitting/Underfitting 딥다이브 문서

학습 과정에서 얻은 경험을 바탕으로 두 가지 버전의 딥다이브 문서를 작성.

### 문서 이력

|      버전      |         분량         |      대상       | 특징                      |
|:------------:|:------------------:|:-------------:|-------------------------|
| **v1 (삭제됨)** |   34KB (A4 10장+)   |    일반 개발자     | 상세했으나 한자 혼용 + 입문자에 비적합  |
| **v2 (최종)**  | **25KB (A4 5~7장)** | **딥러닝 입문 학생** | 한자 제거, 비유 활용, 5개 장으로 압축 |

### v2 문서 구성

|   장   | 제목                          | 핵심 내용                                            |
|:-----:|-----------------------------|--------------------------------------------------|
| **1** | Bias-Variance Trade-off     | 과녁 비유로 이해, 공식 단순화                                |
| **2** | Loss 진동과 오버피팅 전조            | 진동의 3가지 원인, 3가지 빨간 신호                            |
| **3** | Warm-up & Scheduler         | Warm-up 필요성, Cosine Decay, 조합 설계                 |
| **4** | Scaling Laws                | Chinchilla 법칙, "모델 커도 괜찮다"                       |
| **5** | Validation & Early Stopping | patience/delta/restore_best_weights + PyTorch 코드 |

### 수록된 실제 사례

1. Loss plateau를 overfitting으로 오해한 케이스 (v4→v5)
2. 1,200쌍 fine-tuning overfitting (v6)
3. Gradient Norm 오해 (정상 상승 vs 문제 상승)

---

> **작성일:** 2026-06-12
> **총 학습 시간 (A100):** 약 8시간 (v5 22K step + v7 55K step)
> **총 대화 세션:** 3회 (6/1 MoE, 6/2 정리, 6/11~12 Dense+한국어+v7)

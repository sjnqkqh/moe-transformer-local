# MoE Transformer — 학습 결과 보고서

> **모델:** 162.42M MoE Transformer (8층, 4 MoE + 4 Dense interleaved)  
> **데이터:** FineWeb-edu sample-10BT (1M 문서, ~500M 토큰)  
> **하드웨어:** Colab A100 40GB, BF16 mixed precision

---

## 1. 학습 이력

|   버전   | Step  | Block Size | Batch |        LR         |    추가 기법     |  최저 PPL   |  최종 PPL   |
|:------:|:-----:|:----------:|:-----:|:-----------------:|:------------:|:---------:|:---------:|
| **v1** | 5,000 |    512     |   4   |    Cosine 3e-4    |      -       |    155    |  **155**  |
| **v2** | 5,000 |    512     |   4   | Cosine 3e-4 (재시작) | Grad Accum 4 |    60     |  **65**   |
| **v3** | 5,000 |  **1024**  |   4   | Cosine 3e-4 (재시작) | Grad Accum 4 | **33.09** | **49.36** |

> 각 버전은 이전 체크포인트를 불러온 후 LR 재설정하여 5,000 step 추가 학습 (총 누적 15,000 step)

### v1 → v2 → v3 Loss 추이

| 구간                           |   Loss   |    PPL    |               상태               |
|:-----------------------------|:--------:|:---------:|:------------------------------:|
| 학습 전                         |  10.56   |  38,736   |      랜덤 초기화 — 무의미한 문자 생성       |
| v1 완료 (seq=512)              |   5.04   |    155    | Loss 절반 이하로 감소. 기본 문법 패턴 습득 시작 |
| v2 완료 (seq=512, grad_accum)  |   4.17   |    65     |        추가 개선. 어휘 다양성 증가        |
| v3 최고 (seq=1024, grad_accum) | **3.50** | **33.09** |    **단일 최고 기록 (step 4800)**    |
| v3 완료                        |   3.90   | **49.36** |             최종 통과              |

---

## 2. 생성 결과

### 2.1 버전별 출력 비교

**The future of AI is**

|      시점      | 출력                                                                                                                                          |
|:------------:|---------------------------------------------------------------------------------------------------------------------------------------------|
|     학습 전     | `Summary Gayrossryption talked dynam Raj vectors predicts mainalom...`                                                                      |
| v1 (PPL 155) | `the most important part of the world's most important role in the world. The world's most important role...`                               |
| v2 (PPL 65)  | `the only way to get the best of the world's AI systems. The AI system is a powerful tool that can be used to help people with AI systems.` |
| v3 (PPL 49)  | `a very important aspect of the future of AI. The future of AI is a very important aspect of AI. AI is a very important aspect of AI. AI is a very important aspect of AI.`      |

**Machine learning models**

|      시점      | 출력                                                                                                           |
|:------------:|--------------------------------------------------------------------------------------------------------------|
|     학습 전     | `Coalition appraisal afflicted criticalring lungs...`                                                        |
| v1 (PPL 155) | `The first time to learn how to create a new technology...`                                                  |
| v2 (PPL 65)  | `the ability to learn from the classroom is a great way to learn from the classroom.`                        |
| v3 (PPL 49)  | `The first step is to create a model that is used to model the model. The model is used to model the model. The model is used to model the model. The model is used to model the` |

**Once upon a time**

|      시점      | 출력                                                                                                                         |
|:------------:|----------------------------------------------------------------------------------------------------------------------------|
|     학습 전     | `timefire trib CAD proximity fung Commercial mockises...`                                                                  |
| v1 (PPL 155) | `the time of the time was to be the same. The first time the time of the time...`                                          |
| v2 (PPL 65)  | `the first time the first time the first time, the first time...`                                                          |
| v3 (PPL 49)  | `the first thing that the author of the book was the author's name. The author's name was changed to the author's name, and the author's name was changed to the author's name. The author` |

### 한국어 테스트

| 프롬프트 | 출력 |
|:--------:|:-----|
| AI는 현재 우리 삶을 어떻게 | `AI는 현재 우리 삶을 어떻게을 정을 정을 정을 정을 정을 정을 정을 정�` |

> 영어 데이터만 학습(토크나이저 vocab이 영어+ByteLevel 기반)하여 한글 입력은 문자 단위로 분해되지도 않고, 패턴도 전혀 학습되지 않음. 특정 byte 조합을 반복하다가 깨짐.

### 2.2 실질적 평가

```
                    오탈자 없는 단어     문법적 연결      재귀적 반복
                                                         (short context)
학습 전:                 ❌               ❌            -
v1 (PPL 155):          △ 부분            ❌            🔴
v2 (PPL 65):           ✅                △ 부분        🔴
v3 (PPL 49):           ✅                △ 단순 패턴   🔴 (최종까지 미해결)
```

- 오탈자 없는 영어 단어를 나열하는 수준까지 도달했으나, 2~3개 구문 연결 후 곧바로 동일 패턴 반복
- "model is used to model the model", "author's name was changed to the author's name" — **자기 출력을 다시 입력받아 동일 패턴을 증폭**
- 컨텍스트 1024를 학습했지만 생성 결과는 10~15토큰 이내의 단순 패턴만 반복. 컨텍스트 활용 능력은 사실상 없음
- 한국어는 전혀 학습되지 않음. 토크나이저가 ByteLevel 영어 기반이라 한글 입력이 의미 있는 단위로 분해되지 않음

---

## 3. MoE 동작 분석

### 3.1 Load Balancing

| 지표                |         측정값          | 의미               |
|:------------------|:--------------------:|:-----------------|
| Aux Loss (레이어당)   |         2.0          | 이론적 최적값 — 완전 균등  |
| 총 Aux Loss (4레이어) | 8.0 ± 0.05 (전 구간 안정) | 편차 거의 0          |
| Expert Collapse   |        **없음**        | 모든 expert가 지속 활성 |

### 3.2 전문가 역할 분화

| 질문                    | 답변                                                                |
|:----------------------|:------------------------------------------------------------------|
| Expert별 다른 역할을 수행하는가? | **확인 불가. 없을 가능성이 높음.**                                            |
| Aux Loss가 전문성을 유도하는가? | 아니요. 오히려 **모든 expert를 억지로 균등하게 만들어 specialization을 방해**했을 가능성이 큼. |
| Collapse는 막았지만 그 대가로? | 전문가 간 차별화가 사라짐. 4개 expert = 사실상 동일한 FFN 4개 복사본일 수 있음.             |

> Aux Loss가 완벽한 Load Balancing을 달성했지만, 이는 각 expert가 최대한 비슷해지도록 강제한 결과. 전문성 분화(Specialization)가 일어나려면 오히려 어느 정도의 불균형이 필요할
> 수 있음.

---

## 4. PPL 개선 요인 분석

| 요인                                 |  PPL 효과  | 비고                     |
|:-----------------------------------|:--------:|:-----------------------|
| Gradient Accumulation (batch 4→16) | 155→65 ✅ | 수렴 안정화에 효과             |
| Block Size 512→1024                | 65→49 ✅  | 장거리 의존성 학습에 효과         |
| LR Restart                         |  일시적 개선  | 첫 재시작 효과가 두 번째는 미미했음   |
| 추가 Step (5K → 15K 누적)              | 수렴 속도 둔화 | PPL 30 이하로는 추가 개선되지 않음 |

---

## 5. 평가 결과

|        계층         | 지표              |           결과            | 판정 |
|:-----------------:|:----------------|:-----------------------:|:--:|
| **Tier 1** 학습 건전성 | Loss 감소율 63%    |      10.56 → 3.90       | ✅  |
| **Tier 2** 언어 품질  | Validation PPL  | 33.09 (최저) / 49.36 (최종) | ✅  |
| **Tier 3** MoE 건강 | Expert Collapse |         **0건**          | ✅  |
|  **Tier 4** 효율성   | Step 시간 / GPU   |      ~0.3s / ~8GB       | ✅  |
| **Tier 5** 확장 분석  | Expert 프로파일링    |           미수행           | ⬜  |

### 성공 기준 통과 여부

| # | 기준                    |           결과            | 판정 |
|:-:|:----------------------|:-----------------------:|:--:|
| 1 | Step 5000 PPL < 50    | 33.09 (최저) / 49.36 (최종) | ✅  |
| 2 | 모든 Expert 사용률 > 5%    |         25%씩 균등         | ✅  |
| 3 | Load Balance CV < 0.3 |          ≈ 0.0          | ✅  |
| 4 | Expert Collapse 없음    |           0건            | ✅  |

PPL과 Load Balance는 목표를 충족했지만, 생성 품질은 기대치에 크게 못 미쳤음. Looping은 모든 학습 단계에서 해결되지 않았음.

---

## 6. 한계

| 한계                           | 원인                                                         |
|:-----------------------------|------------------------------------------------------------|
| **PPL 30 이하로 개선되지 않음**       | 162M 파라미터의 언어 모델링 능력 한계. 데이터를 더 투입해도 Loss 정체 구간에 도달        |
| **재귀적 반복(Looping) 미해결**      | Argmax decoding 특성상 최고 확률 단어만 반복. Temperature sampling 미도입 |
| **Expert Specialization 없음** | 완벽한 Load Balancing이 오히려 전문성 분화를 억제했을 가능성                   |
| **생성이 "영어 단어 나열" 수준**        | 문장 내 일관성 부족, 주제 유지 불가                                      |

---

## 7. 결론

**수치적으로는 목표를 달성했으나, 생성 품질은 초보적인 수준.**

- PPL을 33까지 낮추고 Loss를 63% 감소시켰지만, 실제 생성 결과는 짧은 문맥에서도 반복적인 재귀 패턴을 벗어나지 못함
- Load Balancing이 완벽하게 작동했으나(8.0 ± 0.05), Expert 간 역할 분화는 확인되지 않음
- 최종 출력 품질: **무작위 문자 → 오탈자 없는 영어 단어 조합** 정도의 발전
- Looping은 최종 버전에서도 동일하게 발생

**"162M MoE Transformer가 학습 자체는 정상적으로 수행되었으나, 의미 있는 자연어 생성 수준에는 도달하지 못했다."**

---

*보고서 작성일: 2026-06-02*

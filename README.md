# Dense Transformer

본 프로젝트는 FineWeb-edu 데이터셋 학습을 위한 162M 파라미터 규모의 Decoder-only Dense Transformer 모델 및 학습/평가 파이프라인 구현체입니다. 로컬 환경(MPS/CPU)에서 로직을 신속히 검증한 후 Google Colab A100(BF16) 환경에서 확장 학습할 수 있도록 설계되었습니다.

---

## 1. 핵심 아키텍처 사양

- **모델 크기:** 약 162.4M 파라미터 (임베딩 및 헤드 가중치 미공유, Untied Embedding 구조)
- **레이어 구성:** 총 12개 레이어 (Dense 레이어)
- **차원 사양:** $d_{model} = 768$, Attention Head 8개 ($d_{head} = 96$), FFN 중간 은닉 차원 $d_{ff} = 3072$
- **입력 제한:** 최대 컨텍스트 윈도우 크기 1024
- **어텐션 및 FFN:** RoPE (Rotary Position Embeddings), 인과적 마스킹(Causal Masking), SwiGLU 활성화 함수 적용
- **정규화 및 정규화기:** Pre-RMSNorm 구조, 어텐션 및 FFN 내부 드롭아웃 적용

---

## 2. 프로젝트 구조

```
├── model/                  # 모델 아키텍처 정의 패키지
│   ├── config.py           # 하이퍼파라미터 설정 클래스 (DenseTransformerConfig)
│   ├── dense_transformer.py# 모델 총조립 및 순전파/역전파 손실 정의
│   ├── transformer_block.py# Dense Transformer 블록
│   ├── attention.py        # Causal Multi-Head Attention (드롭아웃 포함)
│   ├── ffn.py              # SwiGLU 및 Dense FFN
│   ├── rope.py             # Rotary Position Embeddings 주파수 연산 및 적용
│   └── normalization.py    # RMSNorm 레이어
├── docs/                   # 프로젝트 문서 및 설계 사양서 모음
│   ├── DEPENDENCY.md       # 의존성 및 패키지 개발 환경 명세
│   ├── archive/            # MoE 관련 보관 문서
│   │   ├── moe_transformer_blueprint.md
│   │   ├── moe_transformer_todo.md
│   │   └── moe_training_results.md
│   ├── dense-ticklish-flurry.md # Dense 전환 계획 문서
│   ├── 프로젝트_개선_검토_리포트.md # 코드 문제점 분석 및 개선 리포트
│   └── run process.md      # 학습 데이터 전처리 및 구동 가이드
├── tokenizer/              # BPE 토크나이저 학습 스크립트
├── train/                  # 학습, 평가 및 디버깅 파이프라인
│   ├── train.py            # Accelerate 기반 학습 스크립트
│   ├── evaluate.py         # 체크포인트 로드 및 PPL 평가 스크립트
│   ├── local_debug.py      # 로컬 모델 셰이프 및 그래디언트 흐름 검증 스크립트
│   └── utils.py            # 체크포인트 입출력, 로깅 및 NumpyDataset 정의
├── tests/                  # 단위 테스트 디렉토리
└── requirements-local.txt  # 로컬 환경 패키지 요구사항
```

---

## 3. 현재 개발 상태

- **모델 아키텍처 및 파이프라인:** 전체 구현 완료.
- **코드 품질 및 최적화:** 아키텍처 설정을 `DenseTransformerConfig`로 중앙화하고, 어텐션 및 FFN에 드롭아웃을 적용하였습니다.
- **테스트 커버리지:** 설정 호환성, 드롭아웃 동적 변화, 데이터셋 지연 로드 및 순전파/인과적 마스킹 작동을 검증하는 단위 테스트 케이스 구성 완료.

---

## 4. 실행 및 테스트 방법

### 1) 가상환경 및 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

### 2) 단위 테스트 실행
새롭게 작성된 테스트 스크립트들을 통해 구조 개선 사양을 검증합니다.
```bash
python -m unittest discover -s tests
```

### 3) 로컬 디버깅 검증
모델의 빌드 상태, 파라미터 계산 범위 및 역전파 그래디언트 전파 상태를 검사합니다.
```bash
python train/local_debug.py
```

### 4) 로컬 스모크 테스트 (E2E 파이프라인 확인)
토크나이저 학습부터 학습 진행, 최종 평가까지의 전체 루프가 오동작 없이 실행되는지 점검합니다.
```bash
# 1. 토크나이저 학습 스모크 테스트
python tokenizer/train_tokenizer.py --smoke_test --output_dir tokenizer/test_output

# 2. 데이터 전처리 스모크 테스트
python train/prepare_data.py --smoke_test --tokenizer_dir tokenizer/test_output --output_dir train/test_data --block_size 16

# 3. 미니 학습 10스텝 실행
python -m train.train --smoke_test --run_id test_dense --name test_dense --data_dir train/test_data --tokenizer_dir tokenizer/test_output --project_dir test_project --block_size 16 --max_steps 10

# 4. 체크포인트 성능 평가
python -m train.evaluate --smoke_test --ckpt_dir test_project/checkpoints --checkpoint_pattern dense_test_dense --data_dir train/test_data --block_size 16 --output_file test_project/reports/evaluation_report.json
```

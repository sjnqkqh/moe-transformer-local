개인 학습 목적으로 모델의 가중치 구조와 파이프라인 흐름을 면밀히 모니터링할 수 있도록 1) 로깅 출력을 보완하였고, 2) 직접 순서대로 구동해보실 수 있는 가이드를 작성했습니다.

1. 로깅 보완 완료 사항
모델 아키텍처 및 파라미터 정보 시각화 (train/train.py): 학습 스크립트 시작 시점에 모델의 전체 크기뿐만 아니라, 토큰 임베딩, 어텐션 헤드, Dense FFN 및 MoE FFN 전문가군에 속한 구체적인 파라미터 규모를 보기 쉽게 표 형태로 분할 로깅하도록 코드를 보완하였습니다.
데이터 인코딩 진행률 정밀화 (train/prepare_data.py): 기존 5000개 문서 단위의 고정 로깅에서, 처리 대상 크기에 비례하여 10% 단위(10% ~ 100%)로 실시간 비례적 진행 상태가 로깅되도록 업그레이드했습니다.
2. 단계별 로컬 실행 가이드 (터미널에서 직접 실행)
가상환경에 접속한 뒤 아래 명령어를 순서대로 실행하며 중간 로그와 파일을 직접 확인하실 수 있습니다.

[0단계] 가상환경 활성화 및 위치 확인
bash
# 워크스페이스 위치로 이동
cd ~/Desktop/moe-transformer-local
# 가상환경 활성화
source .venv/bin/env/activate  # (또는 .venv/bin/activate)
[1단계] 모델 단독 디버깅 및 그래디언트 흐름 검증
모델 설계의 오류 유무를 체크하고, 역전파 시 가중치 매개변수 전반에 그래디언트가 막힘없이 흐르는지 검증합니다.

명령어:
bash
.venv/bin/python train/local_debug.py
학습 포인트: 로그 최하단에 레이어별(Layer 0, 2, 4, 6) 전문가 4개의 실제 사용률 분배 수치가 고르게 출력되는지 확인해 보세요. (특정 전문가가 0%로 몰리는 'Expert Collapse' 현상이 없음을 검증합니다.)
[2단계] 토크나이저 미니 학습 (스모크 테스트)
데이터 가공 전에 텍스트를 정수 인덱스(Token ID)로 변경해 줄 BPE 토크나이저를 학습합니다.

명령어:
bash
.venv/bin/python tokenizer/train_tokenizer.py --smoke_test --output_dir tokenizer/test_output
학습 포인트: 완료 후 tokenizer/test_output/에 토크나이저 설정 파일들이 생성됩니다. 마지막 출력 로그에서 'Hello, MoE Transformer!...' 문장이 토큰 ID 배열로 인코딩된 뒤 원문으로 안전하게 복원(Decode)되는 흐름을 관찰할 수 있습니다.
[3단계] 데이터셋 인코딩 및 시퀀스 청크 분할
학습에 사용될 텍스트 문서들을 토큰화하여 16 크기의 고정 컨텍스트 윈도우 블록으로 쪼개고, 90:10 비율의 학습/검증 데이터셋으로 나눕니다.

명령어:
bash
.venv/bin/python train/prepare_data.py --smoke_test --tokenizer_dir tokenizer/test_output --output_dir train/test_data --block_size 16
학습 포인트: 새로 업그레이드된 Tokenized XX/200 documents (10.0% ~ 100.0%) 로그 추이를 보며 진행 흐름을 한눈에 알 수 있습니다. 최종 산출물은 train/test_data/train.npy와 val.npy로 저장됩니다.
[4단계] 훈련 루프 실행 (10 스텝 학습)
데이터셋과 토크나이저를 결합해 실제 최적화(Gradient Descent)를 수행하며 가중치를 업데이트합니다.

명령어:
bash
.venv/bin/python -m train.train --smoke_test --run_id test_run --name test_moe --data_dir train/test_data --tokenizer_dir tokenizer/test_output --project_dir test_project --block_size 16 --max_steps 10
학습 포인트:
시작 직후 터미널에 출력되는 **가중치 매개변수 분포 정보(breakdown)**를 통해 모델의 세부 조직 크기를 파악해 보세요.
매 스텝마다 출력되는 Loss가 10점대에서 7점대로 점진적으로 감소하고 Perplexity(PPL)가 작아지는지 확인하세요.
2스텝마다 체크포인트 파일이 저장되는 로그를 모니터링해 보세요.
[5단계] 가중치 복구 및 모델 정밀 평가
학습 완료 후 저장된 10번째 스텝 체크포인트를 불러와 검증 데이터셋에 대한 손실과 전문가 선택 편차 지표(Load Balancing Coefficient of Variation, CV)를 도출합니다.

명령어:
bash
.venv/bin/python -m train.evaluate --smoke_test --ckpt_dir test_project/checkpoints --checkpoint_pattern moe_test_run --data_dir train/test_data --block_size 16 --output_file test_project/reports/evaluation_report.json
학습 포인트:
훈련이 중단되었을 때 체크포인트를 읽어와 가중치를 성공적으로 주입(Load)하는 일련의 순서를 보여줍니다.
전문가 4개 각각의 선택 비중(%)과 할당의 표준 편차 비율인 Load Balancing CV 수치가 0.3 이하로 작아 로드 분산이 성공적인지 확인해 보세요.
최종 성능 통계 보고서가 test_project/reports/evaluation_report.json에 깔끔하게 JSON 포맷으로 누적 저장됩니다.
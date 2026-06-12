import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def prepare_data(
    tokenizer_dir: str,
    output_dir: str,
    num_docs: int = 100000,
    block_size: int = 1024,
    smoke_test: bool = False,
):
    """
    FineWeb-edu 데이터셋의 문서를 토큰 단위 정수 배열로 변환하고
    지정된 block_size(컨텍스트 윈도우 크기) 크기의 훈련/검증 데이터 파일로 가공 및 저장합니다.

    Args:
        tokenizer_dir (str): BPE 토크나이저 파일들이 위치한 폴더 경로.
        output_dir (str): 가공된 train.npy 및 val.npy가 저장될 폴더 경로.
        num_docs (int): 처리할 FineWeb-edu 문서 개수 (기본값 100K).
        block_size (int): 모델의 입력 시퀀스 크기이자 컨텍스트 윈도우 (기본값 1024).
        smoke_test (bool): True일 경우 인터넷 다운로드 없이 미니 데이터셋을 짧은 고정 길이(16)로 처리해 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("📦 Dense Transformer — Dataset Preparation")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # [1단계] 학습용 BPE 토크나이저 복구 로드
    # -------------------------------------------------------------------------
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"    Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # -------------------------------------------------------------------------
    # [2단계] 문서 원본 데이터 로드 (FineWeb-Edu 또는 테스트 데이터)
    # -------------------------------------------------------------------------
    if smoke_test:
        print("Smoke-test mode: Generating dummy documents...")
        # 로컬 테스트용 10개 더미 지문 리스트 생성 (20배 복제해 200문항 생성)
        dummy_texts = [
            "This is a sample document for smoke testing the data preparation script.",
            "Dense transformer uses self-attention and feed-forward networks in every layer.",
            "Each token in the sequence attends to all previous tokens based on causal attention.",
            "Our model has 162M parameters and is trained on FineWeb-edu dataset using PyTorch.",
            "We run training on Colab A100 GPU and use BF16 mixed precision for efficiency.",
            "For local debugging, we run in CPU/MPS mode with float32 precision.",
            "The tokenizer is trained with Byte Pair Encoding (BPE) algorithm.",
            "We have 12 layers of decoder-only dense transformer with standard attention and FFN.",
            "Standard cross-entropy loss is used as the sole training objective for language modeling.",
            "Save checkpoints to Google Drive to handle preemptive runtime terminations.",
        ] * 20
        from datasets import Dataset

        raw_dataset = Dataset.from_dict({"text": dummy_texts})
    else:
        print(
            f"Loading {num_docs} documents from HuggingFaceFW/fineweb-edu (sample-10BT)..."
        )
        # 데이터가 수십 기가 바이트에 달하므로 streaming=True로 한 문서씩 순차 획득합니다.
        stream_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        texts = []
        count = 0
        for item in stream_dataset:
            texts.append(item["text"])
            count += 1
            if count >= num_docs:
                break

        from datasets import Dataset

        raw_dataset = Dataset.from_dict({"text": texts})

    print(f"    Loaded {len(raw_dataset)} documents.")

    # -------------------------------------------------------------------------
    # [3단계] 토큰화 수행 및 단일 거대 토큰 배열로의 병합
    # -------------------------------------------------------------------------
    # 각 문서가 끊기는 지점을 모델에게 알려주기 위해, 문서 사이사이에 End-Of-Sequence(</s>) 토큰 ID를 끼워 넣습니다.
    print("Tokenizing documents...")
    all_tokens = []

    # 토크나이저 사전에서 eos_token_id 값을 조회하고 없으면 bos_token_id로 백업합니다.
    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.bos_token_id
    )

    total_docs = len(raw_dataset)
    # 진행률 출력을 위한 10% 로그 주기 연산
    log_interval = max(1, min(10000, total_docs // 10))

    for i, doc in enumerate(raw_dataset):
        # 문서 텍스트 문자열을 정수(Token ID) 리스트로 인코딩합니다.
        tokens = tokenizer.encode(doc["text"])
        all_tokens.extend(tokens)
        all_tokens.append(eos_token_id)  # 각 문서 끝에 </s> 추가

        # 10% 단위 진행률 출력
        if (i + 1) % log_interval == 0 or (i + 1) == total_docs:
            percentage = ((i + 1) / total_docs) * 100
            print(f"      Tokenized {i + 1}/{total_docs} documents ({percentage:.1f}%)")

    # 전체 정수 리스트를 고성능 배열 연산을 위해 NumPy의 32비트 정수(int32) 어레이로 변환합니다.
    all_tokens = np.array(all_tokens, dtype=np.int32)
    print(f"    Total tokens: {len(all_tokens):,}")

    # -------------------------------------------------------------------------
    # [4단계] block_size 고정 크기로 토큰 스트림 잘라붙이기 (Chunking)
    # -------------------------------------------------------------------------
    # 모델에 집어넣을 데이터는 일정한 시퀀스 길이(block_size)로 차원이 정렬되어야 합니다.
    total_len = len(all_tokens)
    # block_size의 배수가 되도록 자투리 토큰들을 과감하게 잘라 버립니다 (Truncate remainder).
    total_len = (total_len // block_size) * block_size
    if total_len == 0:
        raise ValueError(
            f"토큰 총 개수({len(all_tokens)})가 block_size({block_size}) 1개 분량에도 미치지 못합니다."
        )

    # (Total_Tokens,) 차원의 1차원 평탄 배열을 (N_Blocks, block_size)의 2차원 사각형 구조로 정렬(reshape)합니다.
    blocks = all_tokens[:total_len].reshape(-1, block_size)
    print(f"    Total blocks of size {block_size}: {len(blocks):,}")

    # -------------------------------------------------------------------------
    # [5단계] 훈련 데이터셋과 검증 데이터셋의 분할 (90:10 Split)
    # -------------------------------------------------------------------------
    # 전체 가공 블록 개수의 90% 지점을 기준으로 나눕니다.
    num_blocks = len(blocks)
    split_idx = int(num_blocks * 0.9)
    # 스모크 테스트 시 데이터 수가 너무 적어 검증 셋이 0이 되지 않도록 보정합니다.
    if smoke_test and split_idx == num_blocks:
        split_idx = max(1, num_blocks - 1)

    train_blocks = blocks[:split_idx]
    val_blocks = blocks[split_idx:]

    print(
        f"    Train size: {len(train_blocks):,} blocks ({len(train_blocks) * block_size:,} tokens)"
    )
    print(
        f"    Val size:   {len(val_blocks):,} blocks ({len(val_blocks) * block_size:,} tokens)"
    )

    # -------------------------------------------------------------------------
    # [6단계] NumPy 바이너리 파일(.npy)로 영구 저장
    # -------------------------------------------------------------------------
    # .npy 포맷은 PyTorch DataLoader가 필요할 때 데이터 부분만 효율적으로 파일에서 즉시 읽어 들일 수 있습니다.
    train_path = os.path.join(output_dir, "train.npy")
    val_path = os.path.join(output_dir, "val.npy")

    np.save(train_path, train_blocks)
    np.save(val_path, val_blocks)

    print(f"Saved dataset files:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print("✅ Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/output")
    parser.add_argument("--output_dir", type=str, default="train/data")
    parser.add_argument("--num_docs", type=int, default=100000)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    prepare_data(
        args.tokenizer_dir,
        args.output_dir,
        args.num_docs,
        args.block_size,
        args.smoke_test,
    )

import os
import glob
import json
import numpy as np
import argparse
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def parse_aihub_dialogue(json_path):
    """AI Hub JSON 1개 파일에서 대화들을 추출하여 포맷 문자열 목록으로 반환"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

    sessions = data.get("sessionInfo", [])
    if not sessions:
        # 혹시 단일 세션 포맷이나 예전 키가 있을 경우의 예비 로직
        dialogue = data.get("dialogue", [])
        if dialogue:
            sessions = [{"dialog": dialogue}]
        else:
            return []

    formatted_dialogs = []
    for session in sessions:
        dialog = session.get("dialog", [])
        if not dialog:
            continue

        turns = []
        for turn in dialog:
            # speaker가 "speaker1"인 경우 유저, "speaker2"인 경우 어시스턴트로 매핑
            speaker = str(turn.get("speaker", ""))
            if not speaker:
                speaker = str(turn.get("speaker_id", ""))

            utterance = turn.get("utterance", "")
            if not utterance:
                continue

            # speaker1 -> <user>, speaker2 -> <assistant>
            if speaker == "speaker1" or speaker == "1":
                role = "<user>"
            else:
                role = "<assistant>"

            turns.append(f"{role}{utterance}")

        if turns:
            # 세션 대화 턴들을 <sep>으로 연결하고, 마지막에 EOS 토큰 </s>를 붙임
            formatted_dialogs.append("<sep>".join(turns) + "</s>")

    return formatted_dialogs


def prepare_chat_data(
    data_dir,
    tokenizer_dir,
    output_dir,
    block_size=512,
    smoke_test=False,
    hf_datasets=None,
):
    if hf_datasets is None:
        hf_datasets = []

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("📦 Chatbot Dataset Preparation")
    print("=" * 60)

    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"    Tokenizer loaded. Vocab size: {len(tokenizer)}")

    all_tokens = []

    if smoke_test:
        print("Smoke-test mode: Generating dummy conversational texts...")
        dummy_dialogues = [
            "<user>안녕하세요, 오늘 날씨가 참 좋네요.<sep><assistant>네, 정말 나들이 가기 좋은 날씨예요.</s>",
            "<user>점심 메뉴 추천해주실 수 있나요?<sep><assistant>시원한 냉면이나 돈가스를 드셔보는 건 어떨까요?</s>",
            "<user>오늘 공부를 해야 하는데 집중이 안 돼요.<sep><assistant>잠깐 스트레칭을 하거나 10분 정도 산책을 해보세요.</s>",
            "<user>트랜스포머 모델의 핵심 장점이 무엇인가요?<sep><assistant>멀티 헤드 어텐션을 통한 병렬 처리와 장거리 종속성 학습 능력이 뛰어납니다.</s>",
            "<user>로컬 서버를 띄울 때 fastapi가 좋나요?<sep><assistant>네, fastapi는 빠르고 비동기 처리를 지원하며 자동 문서화가 되어 편리합니다.</s>",
        ] * 40  # 충분한 크기의 토큰을 확보하기 위해 복사
        for text in dummy_dialogues:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

    else:
        # 1. 허깅페이스 데이터셋 처리 (옵션)
        if hf_datasets:
            print(f"Downloading and processing HuggingFace datasets: {hf_datasets}")
            for hf_name in hf_datasets:
                try:
                    print(f"  -> Loading {hf_name}...")
                    dataset = load_dataset(hf_name, split="train")
                    print(f"  -> Found {len(dataset):,} rows in {hf_name}.")

                    for row in dataset:
                        instruction = ""
                        output = ""

                        # 1. Alpaca format
                        if "instruction" in row or "question" in row or "user" in row:
                            instruction = (
                                row.get("instruction")
                                or row.get("question")
                                or row.get("user")
                                or ""
                            )
                            output = (
                                row.get("output")
                                or row.get("answer")
                                or row.get("assistant")
                                or ""
                            )

                        # 2. JaeJiMin/korean_chat_friendly format
                        elif "short_question" in row:
                            instruction = row.get("short_question", "")
                            output = row.get("short_answer", "")

                        # 3. ShareGPT format (conversations)
                        elif "conversations" in row:
                            convs = row["conversations"]
                            # Only handle simple 1-turn for simplicity, or grab the first human/gpt pair
                            if len(convs) >= 2:
                                for c in convs:
                                    if c.get("from") == "human":
                                        instruction = c.get("value", "")
                                    elif c.get("from") in ["gpt", "assistant"]:
                                        output = c.get("value", "")

                        # 4. BAEM1N/nanochat_korean format
                        elif "text" in row:
                            raw_text = row["text"]
                            if "사용자:" in raw_text and "답변:" in raw_text:
                                parts = raw_text.split("답변:")
                                instruction = parts[0].replace("사용자:", "").strip()
                                output = parts[1].strip()

                        if instruction and output:
                            text = f"<user>{instruction}<sep><assistant>{output}</s>"
                            tokens = tokenizer.encode(text)
                            all_tokens.extend(tokens)

                except Exception as e:
                    print(f"  ❌ Failed to load HF dataset {hf_name}: {e}")

        # 2. 로컬 AI Hub 데이터 처리
        if data_dir and os.path.exists(data_dir):
            print(f"Searching for JSON files in {data_dir}...")
        json_paths = sorted(
            glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
        )
        print(f"Found {len(json_paths)} JSON files.")

        total_files = len(json_paths)
        log_interval = max(1, min(1000, total_files // 10))

        for i, json_path in enumerate(json_paths):
            dialogues = parse_aihub_dialogue(json_path)
            for text in dialogues:
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)

            if (i + 1) % log_interval == 0 or (i + 1) == total_files:
                percentage = ((i + 1) / total_files) * 100
                print(f"Processed {i + 1}/{total_files} files ({percentage:.1f}%)")

    if not all_tokens:
        raise ValueError(
            "No tokens were extracted. Please check your data directory or smoke_test status."
        )

    all_tokens = np.array(all_tokens, dtype=np.int32)
    total_tokens = len(all_tokens)
    print(f"Total tokens extracted: {total_tokens:,}")

    total_len = (total_tokens // block_size) * block_size
    if total_len == 0:
        raise ValueError(
            f"Total tokens ({total_tokens}) is less than block_size ({block_size}). Please provide more data or reduce block_size."
        )

    blocks = all_tokens[:total_len].reshape(-1, block_size)
    print(f"Total blocks of size {block_size}: {len(blocks):,}")

    num_blocks = len(blocks)
    split_idx = int(num_blocks * 0.9)
    if split_idx == num_blocks:
        split_idx = max(1, num_blocks - 1)

    train_blocks = blocks[:split_idx]
    val_blocks = blocks[split_idx:]

    print(
        f"Train size: {len(train_blocks):,} blocks ({len(train_blocks) * block_size:,} tokens)"
    )
    print(
        f"Val size:   {len(val_blocks):,} blocks ({len(val_blocks) * block_size:,} tokens)"
    )

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
    parser.add_argument(
        "--data_dir", type=str, default=None, help="로컬 AI Hub 데이터 디렉토리"
    )
    parser.add_argument(
        "--hf_datasets",
        nargs="*",
        default=[],
        help="허깅페이스 데이터셋 이름 목록 (예: Bingsu/KoAlpaca_v1.1a nlpai-lab/kullm-v2)",
    )
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/korean_output")
    parser.add_argument("--output_dir", type=str, default="train/chat_data")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    prepare_chat_data(
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        block_size=args.block_size,
        smoke_test=args.smoke_test,
        hf_datasets=args.hf_datasets,
    )

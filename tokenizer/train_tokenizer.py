import os
import sys
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

def train_tokenizer(output_dir: str, num_docs: int = 50000, smoke_test: bool = False, korean: bool = False, data_dir: str = None, hf_datasets: list = None, hf_samples: int = 10000):
    """
    BPE(Byte-Pair Encoding) 토크나이저를 만들고 저장합니다.
    FineWeb-edu 문서 데이터셋 또는 한국어 대화 데이터셋을 학습에 사용합니다.
    
    Args:
        output_dir (str): 토크나이저 설정 및 사전 파일이 저장될 출력 디렉토리 경로.
        num_docs (int): 영어 모드 시 학습에 동원할 FineWeb-edu 문서 크기 (기본값 50K).
        smoke_test (bool): True일 경우 간이 학습을 진행합니다.
        korean (bool): True일 경우 한국어 대화 토크나이저를 학습합니다.
        data_dir (str): 한국어 대화 JSON 데이터가 있는 디렉토리 경로.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🪙 Transformer — BPE Tokenizer Training")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # [1단계] 학습용 데이터 스트림(Iterator) 준비
    # -------------------------------------------------------------------------
    if korean:
        if smoke_test:
            print("Smoke-test mode activated for Korean. Using mini dataset...")
            texts = [
                "안녕하세요, 오늘 날씨가 참 좋네요.",
                "네, 정말 나들이 가기 좋은 날씨예요.",
                "점심은 맛있게 드셨나요?",
                "네, 비빔밥을 먹었는데 아주 맛있었어요.",
                "어떤 영화를 좋아하시나요?",
                "저는 SF 영화를 즐겨 봅니다.",
                "오늘도 좋은 하루 되세요.",
                "감사합니다. 당신도 좋은 하루 되세요.",
                "파이썬으로 인공지능 모델을 학습시킵니다.",
                "트랜스포머 아키텍처는 혁신적입니다.",
            ] * 10
            iterator = texts
        else:
            if not data_dir:
                raise ValueError("In Korean mode, --data_dir must be specified.")
            print(f"Loading Korean dialog texts from AI-Hub JSONs under {data_dir}...")
            import glob
            import json
            def korean_text_iterator():
                count = 0
                for json_path in glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except Exception as e:
                            print(f"Failed to parse {json_path}: {e}")
                            continue
                    
                    sessions = data.get('sessionInfo', [])
                    if not sessions:
                        dialogue = data.get('dialogue', [])
                        if dialogue:
                            sessions = [{'dialog': dialogue}]
                            
                    for session in sessions:
                        for turn in session.get('dialog', []):
                            utterance = turn.get('utterance', '')
                            if utterance:
                                yield utterance
                                count += 1
                                if count % 100000 == 0:
                                    print(f"Loaded {count} utterances...")
                
                # --- HF 데이터셋 샘플 추가 (토크나이저 vocab 커버리지 확장) ---
                if hf_datasets:
                    from datasets import load_dataset
                    for hf_name in hf_datasets:
                        try:
                            print(f"  Loading HF sample: {hf_name} ({hf_samples} rows)...")
                            ds = load_dataset(hf_name, split="train", streaming=True)
                            sampled = 0
                            for row in ds:
                                text = ""
                                # 다양한 포맷에서 텍스트 추출
                                if "text" in row:
                                    text = row["text"]
                                elif "instruction" in row:
                                    text = row.get("instruction", "") + " " + row.get("output", "") + " " + row.get("answer", "")
                                elif "short_question" in row:
                                    text = row.get("short_question", "") + " " + row.get("short_answer", "")
                                elif "conversations" in row:
                                    parts = [c.get("value", "") for c in row["conversations"] if "value" in c]
                                    text = " ".join(parts)
                                elif "question" in row:
                                    text = row.get("question", "") + " " + row.get("answer", row.get("assistant", ""))
                                elif "user" in row:
                                    text = row.get("user", "") + " " + row.get("assistant", "")
                                if text.strip():
                                    yield text
                                    sampled += 1
                                    if sampled >= hf_samples:
                                        break
                            print(f"    → {sampled} rows yielded from {hf_name}")
                        except Exception as e:
                            print(f"  ❌ Failed to load HF dataset {hf_name}: {e}")
            iterator = korean_text_iterator()
    else:
        if smoke_test:
            print("Smoke-test mode activated. Using mini dataset...")
            texts = [
                "Hello, MoE Transformer!",
                "Mixture of Experts is interesting.",
                "PyTorch makes deep learning easy.",
                "Attention is all you need.",
                "The quick brown fox jumps over the lazy dog.",
                "Deep learning models require large amounts of data.",
                "Natural language processing is a field of AI.",
                "BPE tokenization splits text into subwords.",
                "Colab provides free GPU access for researchers.",
                "Google Drive can be used for storing checkpoints.",
            ] * 10
            iterator = texts
        else:
            print(f"Loading {num_docs} documents from HuggingFaceFW/fineweb-edu (sample-10BT)...")
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            
            def text_iterator():
                count = 0
                for item in dataset:
                    yield item["text"]
                    count += 1
                    if count >= num_docs:
                        break
            iterator = text_iterator()

    # -------------------------------------------------------------------------
    # [2단계] 토크나이저 아키텍처 정의 (BPE + ByteLevel)
    # -------------------------------------------------------------------------
    print("Initializing BPE model and ByteLevel pre-tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # -------------------------------------------------------------------------
    # [3단계] 토크나이저 트레이너 설정 및 어휘 사전 크기 정의
    # -------------------------------------------------------------------------
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    if korean:
        special_tokens += ["<user>", "<assistant>", "<sep>"]
        
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=special_tokens,
        min_frequency=2,
    )
    
    # -------------------------------------------------------------------------
    # [4단계] 학습 실행 및 기본 사전 파일 저장
    # -------------------------------------------------------------------------
    print("Training BPE tokenizer. This may take a few minutes...")
    tokenizer.train_from_iterator(iterator, trainer)
    print("Tokenizer training complete.")
    
    raw_path = os.path.join(output_dir, "raw_bpe_tokenizer.json")
    tokenizer.save(raw_path)
    print(f"Raw BPE tokenizer saved to {raw_path}")
    
    # -------------------------------------------------------------------------
    # [5단계] Hugging Face PreTrainedTokenizerFast 포장 (Transformers 호환)
    # -------------------------------------------------------------------------
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=["<user>", "<assistant>", "<sep>"] if korean else None
    )
    
    fast_tokenizer.save_pretrained(output_dir)
    print(f"PreTrainedTokenizerFast saved to {output_dir}")
    
    # -------------------------------------------------------------------------
    # [6단계] 디코딩 유효성 수동 체크 (Verification)
    # -------------------------------------------------------------------------
    if korean:
        test_text = "<user>안녕하세요<sep><assistant>안녕하세요.</s>"
    else:
        test_text = "Hello, MoE Transformer! Mixture of Experts is awesome."
        
    encoded = fast_tokenizer.encode(test_text)
    decoded = fast_tokenizer.decode(encoded)
    
    print("\nVerification:")
    print(f"  Test text: '{test_text}'")
    print(f"  Encoded IDs: {encoded}")
    print(f"  Decoded text: '{decoded}'")
    assert test_text == decoded, "정밀 복구(Decode) 결과물과 원본이 완전히 일치하지 않습니다!"
    print("✅ Tokenizer verification passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="tokenizer/output")
    parser.add_argument("--num_docs", type=int, default=50000)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--korean", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--hf_datasets", nargs="*", default=[], help="토크나이저 학습에 추가할 HuggingFace 데이터셋 이름 목록 (샘플링됨)")
    parser.add_argument("--hf_samples", type=int, default=10000, help="HF 데이터셋당 샘플링할 row 수 (기본값: 10000)")
    args = parser.parse_args()
    
    train_tokenizer(args.output_dir, args.num_docs, args.smoke_test, args.korean, args.data_dir, args.hf_datasets, args.hf_samples)

import os
import sys
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

def train_tokenizer(output_dir: str, num_docs: int = 50000, smoke_test: bool = False):
    """
    FineWeb-edu 문서 데이터셋을 학습하여 BPE(Byte-Pair Encoding) 토크나이저를 만들고 저장합니다.
    
    Args:
        output_dir (str): 토크나이저 설정 및 사전 파일이 저장될 출력 디렉토리 경로.
        num_docs (int): 토크나이저 학습에 동원할 FineWeb-edu 문서 크기 (기본값 50K).
        smoke_test (bool): True일 경우 인터넷을 타지 않고 내장된 테스트 데이터셋으로 수 초 만에 간이 학습합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🪙 MoE Transformer — BPE Tokenizer Training")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # [1단계] 학습용 데이터 스트림(Iterator) 준비
    # -------------------------------------------------------------------------
    # 토크나이저는 전체 데이터의 글자 빈도수 조합을 관찰해야 하므로, 메모리 절약을 위해 제너레이터(Generator)로 공급합니다.
    if smoke_test:
        print("Smoke-test mode activated. Using mini dataset...")
        # 로컬 실습 및 동작 유효성 통과를 위한 10줄 문장 데이터 (10회 복사하여 총 100줄로 피딩)
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
        # 데이터셋을 전부 다운로드 받지 않고, 필요할 때마다 스트리밍으로 한 문서씩 흘려보내는 streaming=True 기법을 사용합니다.
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        # 제너레이터 함수 정의: num_docs 만큼 문서를 도출하고 멈춥니다.
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
    # BPE 모델 생성: 신조어나 알 수 없는 언어 출현 시 낙오되지 않도록 디폴트 unk_token 지정
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer (사전 토크나이저): 글자 단위 분절 전, 띄어쓰기/공백을 GPT-2 스타일의 바이트 문자(Ġ)로 매핑합니다.
    # add_prefix_space=False는 단어 시작 부분 공백을 별도로 유도하지 않게 처리합니다.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # Decoder (디코더): 토크나이저 출력을 문장으로 디코딩할 때 공백 복원 처리를 자동으로 수행합니다.
    tokenizer.decoder = decoders.ByteLevel()
    
    # -------------------------------------------------------------------------
    # [3단계] 토크나이저 트레이너 설정 및 어휘 사전 크기 정의
    # -------------------------------------------------------------------------
    # vocab_size: 사전 토큰 종류의 개수 (GPT 계열 및 현대 LLaMA류 모델들과 동일하게 32,000으로 고정)
    # special_tokens: 모델 구조가 문장의 시작/끝/패딩 등을 해석하는 데 필요한 제어 토큰 정의
    # min_frequency: 노이즈 분절을 걸러내기 위해 최소 2번 이상 본 단어 조합만 병합 사전에 추가
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=2,
    )
    
    # -------------------------------------------------------------------------
    # [4단계] 학습 실행 및 기본 사전 파일 저장
    # -------------------------------------------------------------------------
    print("Training BPE tokenizer. This may take a few minutes...")
    tokenizer.train_from_iterator(iterator, trainer)
    print("Tokenizer training complete.")
    
    # 토크나이저 본체 구조 단독 저장 (raw json 파일)
    raw_path = os.path.join(output_dir, "raw_bpe_tokenizer.json")
    tokenizer.save(raw_path)
    print(f"Raw BPE tokenizer saved to {raw_path}")
    
    # -------------------------------------------------------------------------
    # [5단계] Hugging Face PreTrainedTokenizerFast 포장 (Transformers 호환)
    # -------------------------------------------------------------------------
    # PyTorch의 배치 처리를 위한 Padding, Attention Mask 생성 등을 위해
    # Hugging Face Transformers에서 제공하는 고속 파이썬 래퍼(Wrapper)로 포장하여 저장합니다.
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    # 최종 사전 파일 및 토크나이저 부속 설정파일 일괄 출력
    fast_tokenizer.save_pretrained(output_dir)
    print(f"PreTrainedTokenizerFast saved to {output_dir}")
    
    # -------------------------------------------------------------------------
    # [6단계] 디코딩 유효성 수동 체크 (Verification)
    # -------------------------------------------------------------------------
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
    args = parser.parse_args()
    
    train_tokenizer(args.output_dir, args.num_docs, args.smoke_test)

import os
import sys
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

def train_tokenizer(output_dir: str, num_docs: int = 50000, smoke_test: bool = False):
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer and saves it in Hugging Face format.
    
    Args:
        output_dir (str): Directory where the tokenizer files will be saved.
        num_docs (int): Number of documents to load from FineWeb-edu for training.
        smoke_test (bool): If True, trains quickly on dummy text for local debugging.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🪙 MoE Transformer — BPE Tokenizer Training")
    print("=" * 60)
    
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
        # Load dataset in streaming mode to avoid downloading the entire 10BT split
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        def text_iterator():
            count = 0
            for item in dataset:
                yield item["text"]
                count += 1
                if count >= num_docs:
                    break
        iterator = text_iterator()

    print("Initializing BPE model and ByteLevel pre-tokenizer...")
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Configure pre-tokenizer and decoder
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Define trainer
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=2,
    )
    
    print("Training BPE tokenizer. This may take a few minutes...")
    tokenizer.train_from_iterator(iterator, trainer)
    print("Tokenizer training complete.")
    
    # Save the base tokenizer file
    raw_path = os.path.join(output_dir, "raw_bpe_tokenizer.json")
    tokenizer.save(raw_path)
    print(f"Raw BPE tokenizer saved to {raw_path}")
    
    # Wrap inside PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    # Save PreTrainedTokenizerFast config files
    fast_tokenizer.save_pretrained(output_dir)
    print(f"PreTrainedTokenizerFast saved to {output_dir}")
    
    # Verify encoding/decoding works
    test_text = "Hello, MoE Transformer! Mixture of Experts is awesome."
    encoded = fast_tokenizer.encode(test_text)
    decoded = fast_tokenizer.decode(encoded)
    
    print("\nVerification:")
    print(f"  Test text: '{test_text}'")
    print(f"  Encoded IDs: {encoded}")
    print(f"  Decoded text: '{decoded}'")
    assert test_text == decoded, "Decoded text does not match the original!"
    print("✅ Tokenizer verification passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="tokenizer/output")
    parser.add_argument("--num_docs", type=int, default=50000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    train_tokenizer(args.output_dir, args.num_docs, args.smoke_test)

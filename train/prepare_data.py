import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

def prepare_data(tokenizer_dir: str, output_dir: str, num_docs: int = 100000, block_size: int = 1024, smoke_test: bool = False):
    """
    Tokenizes FineWeb-edu documents, chunks them into uniform blocks of size block_size,
    splits them into train/val splits (90/10), and saves them as numpy binaries.
    
    Args:
        tokenizer_dir (str): Directory containing the trained tokenizer.
        output_dir (str): Directory where train.npy and val.npy will be saved.
        num_docs (int): Number of documents to preprocess from FineWeb-edu.
        block_size (int): Context window chunk size (1024).
        smoke_test (bool): If True, generates a small dataset for local debugging.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📦 MoE Transformer — Dataset Preparation")
    print("=" * 60)
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"    Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # 2. Load Documents
    if smoke_test:
        print("Smoke-test mode: Generating dummy documents...")
        dummy_texts = [
            "This is a sample document for smoke testing the data preparation script.",
            "Mixture of experts (MoE) uses a router to forward inputs to selected experts.",
            "Each token in the sequence gets assigned to top-K experts based on routing probability.",
            "Our model has 120M parameters and is trained on FineWeb-edu dataset using PyTorch.",
            "We run training on Colab A100 GPU and use BF16 mixed precision for efficiency.",
            "For local debugging, we run in CPU/MPS mode with float32 precision.",
            "The tokenizer is trained with Byte Pair Encoding (BPE) algorithm.",
            "We have 8 layers of decoder-only transformer with interleaved MoE and Dense layers.",
            "Load balancing loss and router z-loss are added to the main training loss.",
            "Save checkpoints to Google Drive to handle preemptive runtime terminations."
        ] * 20
        from datasets import Dataset
        raw_dataset = Dataset.from_dict({"text": dummy_texts})
    else:
        print(f"Loading {num_docs} documents from HuggingFaceFW/fineweb-edu (sample-10BT)...")
        stream_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
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
    
    # 3. Tokenize and Concatenate
    print("Tokenizing documents...")
    all_tokens = []
    
    # Separate documents with end-of-sequence token
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    
    total_docs = len(raw_dataset)
    # Print progress every 10% of total docs, capped at max every 10k docs, min every 1 doc
    log_interval = max(1, min(10000, total_docs // 10))
    
    for i, doc in enumerate(raw_dataset):
        tokens = tokenizer.encode(doc["text"])
        all_tokens.extend(tokens)
        all_tokens.append(eos_token_id)
        if (i + 1) % log_interval == 0 or (i + 1) == total_docs:
            percentage = ((i + 1) / total_docs) * 100
            print(f"      Tokenized {i + 1}/{total_docs} documents ({percentage:.1f}%)")
            
    all_tokens = np.array(all_tokens, dtype=np.int32)
    print(f"    Total tokens: {len(all_tokens):,}")
    
    # 4. Chunk into Blocks of block_size
    # Truncate remainder to make full blocks
    total_len = len(all_tokens)
    total_len = (total_len // block_size) * block_size
    if total_len == 0:
        raise ValueError(f"Not enough tokens ({len(all_tokens)}) to form a block of size {block_size}.")
        
    blocks = all_tokens[:total_len].reshape(-1, block_size)
    print(f"    Total blocks of size {block_size}: {len(blocks):,}")
    
    # 5. Train/Val Split (90:10)
    num_blocks = len(blocks)
    split_idx = int(num_blocks * 0.9)
    if smoke_test and split_idx == num_blocks:
        split_idx = max(1, num_blocks - 1)
        
    train_blocks = blocks[:split_idx]
    val_blocks = blocks[split_idx:]
    
    print(f"    Train size: {len(train_blocks):,} blocks ({len(train_blocks) * block_size:,} tokens)")
    print(f"    Val size:   {len(val_blocks):,} blocks ({len(val_blocks) * block_size:,} tokens)")
    
    # Save as numpy binary files
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
    
    prepare_data(args.tokenizer_dir, args.output_dir, args.num_docs, args.block_size, args.smoke_test)

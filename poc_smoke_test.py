"""
moe-transformer-local — 의존성 PoC (Smoke Test)
로컬 M2 Pro CPU 모드에서 모든 import와 기본 연산이 정상 동작하는지 검증
"""
import sys, platform, math

print("=" * 60)
print("🔥 MoE Transformer — Dependency PoC")
print("=" * 60)

# --- 1. Python + System ---
print(f"\n[1/6] System: {platform.platform()}")
print(f"    Python: {sys.version.split()[0]}")
print(f"    Arch:   {platform.machine()}")

# --- 2. PyTorch ---
print(f"\n[2/6] PyTorch")
import torch
print(f"    Version: {torch.__version__}")
print(f"    CPU cores: {torch.get_num_threads()}")
print(f"    MPS available: {torch.backends.mps.is_available()}")
print(f"    MPS built: {torch.backends.mps.is_built()}")

# 간단한 텐서 연산
x = torch.randn(2, 768)
y = torch.randn(768, 2048)
z = x @ y
assert z.shape == (2, 2048), f"matmul failed: {z.shape}"
print(f"    ✅ matmul (2,768) @ (768,2048) = {list(z.shape)}")

# RMSNorm 수동 검증
rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
normed = x / rms
print(f"    ✅ RMSNorm pass (mean={normed.mean().item():.4f}, std={normed.std().item():.4f})")

# --- 3. HuggingFace Tokenizers ---
print(f"\n[3/6] Tokenizers (BPE 학습)")
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

bpe = Tokenizer(models.BPE(unk_token="<unk>"))
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    min_frequency=2,
)

# 미니 데이터로 학습 (PoC용)
mini_texts = [
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
]

bpe.train_from_iterator(mini_texts, trainer)
print(f"    ✅ BPE trained on {len(mini_texts)} texts")
print(f"    ✅ Vocab size: {bpe.get_vocab_size()}")

# --- 4. HuggingFace Transformers ---
print(f"\n[4/6] Transformers (Tokenizer Wrapper)")
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=bpe,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

encoded = hf_tokenizer("Hello, MoE Transformer!")
tokens = hf_tokenizer.convert_ids_to_tokens(encoded["input_ids"])
print(f"    ✅ Encode: {encoded['input_ids']}")
print(f"    ✅ Tokens: {tokens}")

# --- 5. Datasets ---
print(f"\n[5/6] Datasets")
from datasets import Dataset, Features, Value

dummy_data = Dataset.from_dict({
    "input_ids": [encoded["input_ids"] * 4],  # 반복으로 길이 늘림 (block_size=16 정도)
    "labels": [encoded["input_ids"] * 4],
})
print(f"    ✅ Dataset created: {len(dummy_data)} samples")
print(f"    ✅ Features: {dummy_data.features}")
print(f"    ✅ Sample length: {len(dummy_data[0]['input_ids'])}")

# --- 6. Accelerate ---
print(f"\n[6/6] Accelerate")
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="no")
print(f"    ✅ Accelerator: device={accelerator.device}")
print(f"    ✅ Mixed precision: {accelerator.mixed_precision}")

# Accelerate로 model/dataloader 준비
from torch.utils.data import DataLoader
dummy_loader = DataLoader(dummy_data, batch_size=2)
model = torch.nn.Linear(768, 32000)  # 간단한 LM head 대용
model, loader = accelerator.prepare(model, dummy_loader)
print(f"    ✅ Accelerator prepare OK")

print("\n" + "=" * 60)
import importlib.metadata as meta
print("🎯 ALL DEPENDENCY CHECKS PASSED")
print(f"    torch={torch.__version__}")
print(f"    transformers={meta.version('transformers')}")
print(f"    accelerate={meta.version('accelerate')}")
print(f"    datasets={meta.version('datasets')}")
print(f"    tokenizers={meta.version('tokenizers')}")
print("=" * 60)

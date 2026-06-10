# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements-local.txt   # or: pip install -r requirements-local.txt

# Unit tests
python -m unittest discover -s tests

# Run a single test file
python -m unittest tests.test_forward

# Local shape/gradient validation (no data required)
python train/local_debug.py

# End-to-end smoke test (runs entire pipeline with tiny data)
python tokenizer/train_tokenizer.py --smoke_test --output_dir tokenizer/test_output
python train/prepare_data.py --smoke_test --tokenizer_dir tokenizer/test_output --output_dir train/test_data --block_size 16
python -m train.train --smoke_test --run_id test_dense --name test_dense --data_dir train/test_data --tokenizer_dir tokenizer/test_output --project_dir test_project --block_size 16 --max_steps 10
python -m train.evaluate --smoke_test --ckpt_dir test_project/checkpoints --checkpoint_pattern dense_test_dense --data_dir train/test_data --block_size 16 --output_file test_project/reports/evaluation_report.json
```

## Architecture

This is a 162M-parameter **Decoder-only Dense Transformer** trained on FineWeb-edu. The design targets local logic verification (MPS/CPU) then scales to Google Colab A100 (BF16).

### Model structure (`model/`)

The model consists of 12 identical `TransformerBlock` layers (`n_layers=12`, `d_ff=3072`, `d_model=768`).
All hyperparameters flow through `DenseTransformerConfig` (`model/config.py`). The constructor accepts both a config object and legacy `**kwargs` for backward compatibility.

Key design choices:
- **Untied embeddings**: `token_embeddings` and `lm_head` weights are separate (not shared)
- **Pre-RMSNorm**: normalization applied before attention/FFN, plus a final norm before the LM head
- **RoPE**: frequencies precomputed once at init, stored as a non-persistent buffer (`freqs_cis`), sliced to actual sequence length each forward pass
- **Causal shift in `forward()`**: `DenseTransformer.forward()` performs the `shift_logits / shift_labels` offset internally — callers pass `labels=input_ids` directly

### Training pipeline (`train/`)

- `prepare_data.py` → tokenizes and saves data as `.npy` files (memory-mapped at load time via `NumpyDataset`)
- `train.py` → Accelerate-based training loop. Logs main loss, total loss, learning rate, ppl, speed, gradient norm, and GPU memory usage.
- `evaluate.py` → loads checkpoint, computes perplexity
- `utils.py` → shared `NumpyDataset`, checkpoint save/load, JSONL metrics logging

Checkpoints are named `dense_{run_id}_step{N}.pt` and contain model, optimizer, and scheduler state. Resume is automatic — `load_latest_checkpoint` finds the most recently modified file matching the pattern.

Local smoke test forces `mixed_precision="no"` (FP32). Colab runs use `mixed_precision="bf16"`.

### Training monitoring

Metrics are appended to `metrics_{run_id}.jsonl`. Events (such as loss spikes) are written as individual JSON files under `{project_dir}/logs/events/`.

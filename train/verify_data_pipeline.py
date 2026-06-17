"""
verify_data_pipeline.py — 데이터셋 sanity test

각 HF 데이터셋에서 N rows만 받아 다음을 검증:
  1. 데이터셋 로딩 (config 이름 정확한지)
  2. extract_text_from_row가 유효 텍스트를 뽑아내는지
  3. tokenizer가 한국어를 제대로 처리하는지
  4. 첫 sample 미리보기 (text + token IDs + decode 결과)
  5. 데이터셋별 평균 토큰 길이 / 유효 row 비율

실행: 1-3분 (각 데이터셋 100 rows × 4-5개 데이터셋)
출력: {output_dir}/verify_train.npy + verify_summary.json
"""

import os
import sys
import json
import argparse
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_chat_data import (
    should_stream,
    extract_text_from_row,
)


def verify_one_dataset(hf_name, hf_config, tokenizer, n_samples, preview_chars=200):
    """한 데이터셋의 첫 N rows를 받아 통계와 샘플을 반환."""
    dataset_id = hf_name + (f":{hf_config}" if hf_config else "")
    streaming = should_stream(hf_name)
    print(f"\n{'─' * 60}")
    print(f"📥 {dataset_id}" + (" [STREAMING]" if streaming else ""))
    print(f"   샘플 {n_samples} rows 추출 중...")

    summary = {
        "dataset_id": dataset_id,
        "streaming": streaming,
        "load_success": False,
        "rows_seen": 0,
        "rows_with_text": 0,
        "total_tokens": 0,
        "avg_tokens_per_row": 0.0,
        "first_sample": None,
        "error": None,
    }

    try:
        ds = load_dataset(hf_name, hf_config, split="train", streaming=streaming)
        summary["load_success"] = True
    except Exception as e:
        summary["error"] = f"load_dataset 실패: {e}"
        print(f"   ❌ {summary['error']}")
        return summary, []

    texts = []
    raw_samples = []
    for row in ds:
        summary["rows_seen"] += 1
        text = extract_text_from_row(row)
        if text:
            summary["rows_with_text"] += 1
            texts.append(text)
            if summary["first_sample"] is None:
                summary["first_sample"] = {
                    "raw_keys": list(row.keys()) if hasattr(row, "keys") else None,
                    "extracted_text_preview": text[:preview_chars] + ("..." if len(text) > preview_chars else ""),
                    "extracted_text_length": len(text),
                }
            raw_samples.append(text)
        if summary["rows_seen"] >= n_samples:
            break

    if not texts:
        summary["error"] = "유효 텍스트 0개 (extract_text_from_row가 인식하지 못함)"
        print(f"   ❌ {summary['error']}")
        return summary, []

    # 배치 토크나이즈
    enc = tokenizer(texts, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
    token_lengths = [len(ids) for ids in enc if ids]
    summary["total_tokens"] = sum(token_lengths)
    summary["avg_tokens_per_row"] = summary["total_tokens"] / len(token_lengths) if token_lengths else 0

    # 첫 sample의 토큰 정보 추가
    if enc and enc[0]:
        first_ids = enc[0][:50]  # 처음 50개 토큰만
        first_decoded = tokenizer.decode(first_ids)
        summary["first_sample"]["first_50_token_ids"] = first_ids
        summary["first_sample"]["first_50_decoded"] = first_decoded

    print(f"   ✅ rows_seen={summary['rows_seen']}, "
          f"rows_with_text={summary['rows_with_text']} "
          f"({summary['rows_with_text']/summary['rows_seen']*100:.0f}%)")
    print(f"   📊 total_tokens={summary['total_tokens']:,}, "
          f"avg={summary['avg_tokens_per_row']:.0f} tok/row")
    if summary["first_sample"]:
        fs = summary["first_sample"]
        print(f"   📝 첫 sample ({fs['extracted_text_length']} chars):")
        print(f"      \"{fs['extracted_text_preview']}\"")
        print(f"   🔢 첫 50 tokens decoded:")
        print(f"      \"{fs.get('first_50_decoded', '')}\"")

    # 토큰화한 ids를 합쳐서 반환 (npy 저장용)
    flat = []
    for ids in enc:
        if ids:
            flat.extend(ids)
    return summary, flat


def verify(args):
    print("=" * 60)
    print("🔬 Data Pipeline Sanity Test")
    print(f"   datasets   : {len(args.hf_datasets)} 개")
    print(f"   n_samples  : {args.n_samples} rows / dataset")
    print(f"   output_dir : {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    print(f"   vocab_size={len(tokenizer)}")

    all_summaries = []
    all_tokens = []

    for entry in args.hf_datasets:
        hf_name, _, hf_config = entry.partition(":")
        hf_config = hf_config or None
        summary, tokens = verify_one_dataset(
            hf_name, hf_config, tokenizer,
            n_samples=args.n_samples,
            preview_chars=args.preview_chars,
        )
        all_summaries.append(summary)
        all_tokens.extend(tokens)

    # 요약 저장
    summary_path = os.path.join(args.output_dir, "verify_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_datasets": len(all_summaries),
            "successful_datasets": sum(1 for s in all_summaries if s["load_success"] and s["total_tokens"] > 0),
            "total_tokens_collected": sum(s["total_tokens"] for s in all_summaries),
            "datasets": all_summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📄 요약 저장: {summary_path}")

    # 토큰 .npy 저장 (block_size로 reshape)
    if all_tokens and args.save_npy:
        arr = np.asarray(all_tokens, dtype=np.int32)
        total_len = (len(arr) // args.block_size) * args.block_size
        if total_len >= args.block_size:
            blocks = arr[:total_len].reshape(-1, args.block_size)
            npy_path = os.path.join(args.output_dir, "verify_train.npy")
            np.save(npy_path, blocks)
            print(f"💾 verify_train.npy 저장: {len(blocks):,} blocks × {args.block_size} tokens "
                  f"({total_len:,} tokens, {os.path.getsize(npy_path)/1e6:.1f} MB)")
        else:
            print(f"⚠️ 토큰 수({len(arr)}) < block_size({args.block_size}) → .npy 저장 스킵")

    # 최종 판정
    print("\n" + "=" * 60)
    print("📊 최종 판정")
    print("=" * 60)
    failed = [s["dataset_id"] for s in all_summaries if not s["load_success"] or s["total_tokens"] == 0]
    if failed:
        print(f"❌ 실패 데이터셋 ({len(failed)}):")
        for d in failed:
            err = next(s["error"] for s in all_summaries if s["dataset_id"] == d)
            print(f"   - {d}: {err}")
    else:
        print(f"✅ 모든 데이터셋 통과! ({len(all_summaries)}/{len(all_summaries)})")
    print(f"📦 총 수집 토큰: {sum(s['total_tokens'] for s in all_summaries):,}")
    print("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_datasets", nargs="+", required=True,
                        help="검증할 HF 데이터셋. 'name' 또는 'name:config' 형식.")
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="train/verify_data")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="데이터셋당 추출 row 수 (기본 100)")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--preview_chars", type=int, default=200,
                        help="텍스트 미리보기 길이")
    parser.add_argument("--save_npy", action="store_true",
                        help="verify_train.npy 저장 여부")
    args = parser.parse_args()
    sys.exit(verify(args))

"""
prepare_chat_data.py v8.2 — 세션 재시작 안전 / chunk 분리 / 재개 가능

저장 전략:
  - chunks_dir (Drive 영구 저장): chunk .bin 파일 + _progress.json
    sequential 2GB 쓰기 + fsync → Drive FUSE에서도 안전
    Colab 세션이 죽어도 다음 세션에서 그대로 재개 가능
  - assembly_scratch (로컬 SSD, 휘발성): train.npy/val.npy 조립 작업용
    memmap-based random-access write가 필요하므로 로컬에서만 처리
    조립 끝나면 최종 .npy 파일을 Drive로 sequential copy

크래시 복구:
  - chunk 쓰는 중 죽음 → 미커밋 chunk는 진행상태에 없음, 다음 row부터 재시작
  - chunk 커밋 후 죽음 → progress.json에 기록됨, 다음 dataset 또는 chunk부터 재시작
  - 조립 중 죽음 → chunk는 남아 있음, 다음 실행에서 조립 단계만 재시도
"""

import os
import glob
import json
import shutil
import hashlib
import tempfile
import argparse
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


STREAMING_KEYWORDS = ("fineweb", "culturax", "mc4", "oscar", "cc100", "/c4")
DEFAULT_CHUNK_TOKENS = 500_000_000  # 500M tokens/chunk ≈ 2GB


# ── 경로 / 디스크 / 진행상태 ────────────────────────────────────────────────

def should_stream(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in STREAMING_KEYWORDS)


def resolve_chunks_dir(output_dir: str, explicit: str = None) -> str:
    """
    chunk 파일 영구 저장 위치. Drive 안에 저장해 세션 재시작 후에도 살아남음.
    sequential write + fsync 가 모두 chunk 단위(2GB)라 Drive FUSE에서도 안전.
    """
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    chunks = os.path.join(output_dir, "_chunks_persistent")
    os.makedirs(chunks, exist_ok=True)
    return chunks


def resolve_assembly_scratch(output_dir: str, explicit: str = None) -> str:
    """
    조립용 임시 디렉토리. memmap random-access가 필요하므로 로컬 SSD에 둠.
    train.npy/val.npy 임시본만 들어가고 끝나면 정리됨.
    """
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    if os.path.isdir("/content") and output_dir.startswith("/content/drive/"):
        s = "/content/_prepare_chat_assembly"
        os.makedirs(s, exist_ok=True)
        return s
    return tempfile.mkdtemp(prefix="prepare_chat_asm_")


def check_disk_space(path: str, required_bytes: int) -> None:
    stat = shutil.disk_usage(path)
    if stat.free < required_bytes:
        raise RuntimeError(
            f"❌ 디스크 부족: {path}\n"
            f"   필요: {required_bytes/1e9:.1f} GB / 가용: {stat.free/1e9:.1f} GB\n"
            f"   여유 공간 확보 후 재시도."
        )
    print(f"💾 디스크: {path} ({stat.free/1e9:.1f} GB 가용, {required_bytes/1e9:.1f} GB 필요)")


def config_hash(*items) -> str:
    s = json.dumps(items, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def load_progress(chunks_dir: str):
    p = os.path.join(chunks_dir, "_progress.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_progress(chunks_dir: str, progress: dict) -> None:
    p = os.path.join(chunks_dir, "_progress.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # Drive FUSE에서 fsync 미지원이면 무시
    os.replace(tmp, p)


def cleanup_chunks_dir(chunks_dir: str) -> None:
    if not os.path.isdir(chunks_dir):
        return
    for fname in os.listdir(chunks_dir):
        if fname.startswith("_chunk_") or fname == "_progress.json":
            try:
                os.remove(os.path.join(chunks_dir, fname))
            except Exception:
                pass


def cleanup_assembly(assembly_dir: str) -> None:
    if not os.path.isdir(assembly_dir):
        return
    for fname in ("train.npy", "val.npy"):
        try:
            os.remove(os.path.join(assembly_dir, fname))
        except Exception:
            pass


# ── 데이터셋 row → 학습 텍스트 ──────────────────────────────────────────────

def parse_aihub_dialogue(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    sessions = data.get("sessionInfo") or ([{"dialog": data.get("dialogue", [])}] if data.get("dialogue") else [])
    out = []
    for s in sessions:
        turns = []
        for t in s.get("dialog", []):
            speaker = str(t.get("speaker", "") or t.get("speaker_id", ""))
            utt = t.get("utterance", "")
            if not utt:
                continue
            role = "<user>" if speaker in ("speaker1", "1") else "<assistant>"
            turns.append(f"{role}{utt}")
        if turns:
            out.append("<sep>".join(turns) + "</s>")
    return out


def extract_text_from_row(row) -> str:
    instr = row.get("instruction") or row.get("question") or row.get("user") or ""
    out = row.get("output") or row.get("answer") or row.get("assistant") or ""
    if instr and out:
        return f"<user>{instr}<sep><assistant>{out}</s>"

    convs = row.get("conversations")
    if isinstance(convs, list) and convs:
        parts = []
        for c in convs:
            role = c.get("from", "")
            val = c.get("value", "")
            if not val:
                continue
            if role == "human":
                parts.append(f"<user>{val}")
            elif role in ("gpt", "assistant"):
                parts.append(f"<assistant>{val}")
        if parts:
            return "<sep>".join(parts) + "</s>"

    if "short_question" in row:
        q, a = row.get("short_question", ""), row.get("short_answer", "")
        if q and a:
            return f"<user>{q}<sep><assistant>{a}</s>"

    text = row.get("text") or row.get("content") or ""
    if isinstance(text, str):
        text = text.strip()
        if len(text) >= 50:
            if "사용자:" in text and "답변:" in text:
                try:
                    p = text.split("답변:", 1)
                    instr = p[0].replace("사용자:", "").strip()
                    ans = p[1].strip()
                    if instr and ans:
                        return f"<user>{instr}<sep><assistant>{ans}</s>"
                except Exception:
                    pass
            return text
    return ""


# ── 한 데이터셋 → chunk 파일들 (Drive 영구 저장) ────────────────────────────

def process_dataset_into_chunks(
    hf_name, hf_config, tokenizer, chunks_dir, progress,
    target_tokens, chunk_max_tokens, max_samples_per_dataset, batch_size
) -> int:
    dataset_id = hf_name + (f":{hf_config}" if hf_config else "")
    ds_state = progress["datasets"].setdefault(dataset_id, {
        "samples_consumed": 0, "tokens_emitted": 0, "complete": False,
    })

    if ds_state["complete"]:
        print(f"\n⏭  {dataset_id}: 이미 완료 ({ds_state['tokens_emitted']:,} tokens) — 스킵")
        return 0

    streaming = should_stream(hf_name)
    skip_n = ds_state["samples_consumed"]
    print(f"\n📥 {dataset_id}" + (" [STREAMING]" if streaming else "") +
          (f" | resume: skip {skip_n:,} rows" if skip_n else ""))

    try:
        dataset = load_dataset(hf_name, hf_config, split="train", streaming=streaming)
        if skip_n > 0:
            if streaming:
                dataset = dataset.skip(skip_n)
            else:
                if skip_n < len(dataset):
                    dataset = dataset.select(range(skip_n, len(dataset)))
                else:
                    ds_state["complete"] = True
                    save_progress(chunks_dir, progress)
                    return 0
    except Exception as e:
        print(f"   ❌ load 실패: {e}")
        return 0

    chunk_idx = len(progress["chunks"]) + 1
    chunk_path = os.path.join(chunks_dir, f"_chunk_{chunk_idx:04d}.bin")
    chunk_fout = open(chunk_path, "wb")
    chunk_tokens = 0
    buffer_texts = []
    samples_added = 0
    tokens_added = 0

    def flush_buffer():
        nonlocal chunk_tokens, tokens_added
        if not buffer_texts:
            return
        enc = tokenizer(
            buffer_texts, add_special_tokens=False,
            truncation=False, return_attention_mask=False,
        )["input_ids"]
        for ids in enc:
            if ids:
                arr = np.asarray(ids, dtype=np.int32)
                arr.tofile(chunk_fout)
                chunk_tokens += len(ids)
                tokens_added += len(ids)
        buffer_texts.clear()

    def commit_chunk_and_rotate():
        nonlocal chunk_idx, chunk_path, chunk_fout, chunk_tokens
        chunk_fout.flush()
        try:
            os.fsync(chunk_fout.fileno())
        except OSError:
            pass
        chunk_fout.close()

        progress["chunks"].append({
            "file": os.path.basename(chunk_path),
            "tokens": chunk_tokens,
            "dataset": dataset_id,
            "chunk_idx": chunk_idx,
        })
        ds_state["samples_consumed"] = skip_n + samples_added
        ds_state["tokens_emitted"] += chunk_tokens
        progress["total_tokens"] += chunk_tokens
        save_progress(chunks_dir, progress)
        print(f"   💾 chunk #{chunk_idx} → Drive 저장: {chunk_tokens:,} tokens "
              f"| total={progress['total_tokens']:,}")

        chunk_idx = len(progress["chunks"]) + 1
        chunk_path = os.path.join(chunks_dir, f"_chunk_{chunk_idx:04d}.bin")
        chunk_fout = open(chunk_path, "wb")
        chunk_tokens = 0

    try:
        for row in dataset:
            if target_tokens and progress["total_tokens"] + tokens_added >= target_tokens:
                break
            if max_samples_per_dataset and (skip_n + samples_added) >= max_samples_per_dataset:
                break

            text = extract_text_from_row(row)
            if not text:
                continue
            buffer_texts.append(text)
            samples_added += 1

            if len(buffer_texts) >= batch_size:
                flush_buffer()
                if chunk_tokens >= chunk_max_tokens:
                    commit_chunk_and_rotate()
                if samples_added % 10000 == 0:
                    total_so_far = progress["total_tokens"] + tokens_added
                    pct = (total_so_far * 100 / target_tokens) if target_tokens else 0
                    print(f"   {samples_added:,} samples | +{tokens_added:,} tok "
                          f"| total {total_so_far:,} ({pct:.1f}%)")
        flush_buffer()
    except Exception as e:
        print(f"   ⚠️ 스트리밍 중 오류 (기존 chunk 보존됨): {e}")

    if chunk_tokens > 0:
        commit_chunk_and_rotate()
    else:
        chunk_fout.close()
        try:
            os.remove(chunk_path)
        except Exception:
            pass

    ds_state["complete"] = True
    save_progress(chunks_dir, progress)
    print(f"   ✅ {dataset_id}: +{samples_added:,} samples, +{tokens_added:,} tokens")
    return tokens_added


# ── 최종 .npy 조립 (chunks=Drive → npy=로컬→Drive) ───────────────────────

def assemble_chunks_into_npy(chunks_dir, assembly_scratch, output_dir, block_size, progress):
    chunks = progress["chunks"]
    if not chunks:
        raise ValueError("저장된 chunk 없음.")

    total_tokens = sum(c["tokens"] for c in chunks)
    total_blocks = total_tokens // block_size
    if total_blocks == 0:
        raise ValueError(f"수집 토큰({total_tokens}) < block_size({block_size})")

    split_idx = max(1, int(total_blocks * 0.95))
    train_blocks = split_idx
    val_blocks = total_blocks - split_idx

    print(f"\n📐 Assembling {len(chunks)} chunks → "
          f"train {train_blocks:,} / val {val_blocks:,} blocks (block_size={block_size})")
    print(f"   chunks   : {chunks_dir}")
    print(f"   assembly : {assembly_scratch} (로컬 SSD)")

    train_scratch = os.path.join(assembly_scratch, "train.npy")
    val_scratch = os.path.join(assembly_scratch, "val.npy")
    train_mm = np.lib.format.open_memmap(
        train_scratch, mode="w+", dtype=np.int32, shape=(train_blocks, block_size)
    )
    val_mm = np.lib.format.open_memmap(
        val_scratch, mode="w+", dtype=np.int32, shape=(val_blocks, block_size)
    )

    written_train = 0
    written_val = 0
    leftover = np.zeros(0, dtype=np.int32)

    for c in chunks:
        chunk_path = os.path.join(chunks_dir, c["file"])
        if not os.path.exists(chunk_path):
            print(f"   ⚠️ 누락 chunk 스킵: {c['file']}")
            continue
        chunk_data = np.fromfile(chunk_path, dtype=np.int32)
        if leftover.size > 0:
            chunk_data = np.concatenate([leftover, chunk_data])
        n_full = chunk_data.size // block_size
        n_used = n_full * block_size
        blocks = (chunk_data[:n_used].reshape(n_full, block_size)
                  if n_full > 0 else np.zeros((0, block_size), dtype=np.int32))
        leftover = chunk_data[n_used:].copy() if n_used < chunk_data.size else np.zeros(0, dtype=np.int32)

        if written_train < train_blocks:
            n_to = min(train_blocks - written_train, n_full)
            train_mm[written_train:written_train + n_to] = blocks[:n_to]
            written_train += n_to
            blocks = blocks[n_to:]
            n_full -= n_to

        if n_full > 0 and written_val < val_blocks:
            n_to = min(val_blocks - written_val, n_full)
            val_mm[written_val:written_val + n_to] = blocks[:n_to]
            written_val += n_to

        if written_train >= train_blocks and written_val >= val_blocks:
            break

    train_mm.flush()
    val_mm.flush()
    del train_mm, val_mm

    train_path = os.path.join(output_dir, "train.npy")
    val_path = os.path.join(output_dir, "val.npy")
    print(f"\n📤 scratch → Drive sequential copy:")
    print(f"   train.npy ({os.path.getsize(train_scratch) / 1e9:.2f} GB)")
    shutil.copy2(train_scratch, train_path)
    print(f"   val.npy   ({os.path.getsize(val_scratch) / 1e9:.2f} GB)")
    shutil.copy2(val_scratch, val_path)

    print(f"\n✅ train.npy: {train_blocks:,} blocks ({train_blocks * block_size:,} tokens)")
    print(f"✅ val.npy  : {val_blocks:,} blocks ({val_blocks * block_size:,} tokens)")
    return train_path, val_path


# ── 메인 ──────────────────────────────────────────────────────────────────

def prepare_chat_data(
    data_dir, tokenizer_dir, output_dir,
    block_size=2048, smoke_test=False,
    hf_datasets=None, target_tokens=None,
    max_samples_per_dataset=None, batch_size=1000,
    chunks_dir=None, assembly_scratch=None,
    chunk_size_tokens=DEFAULT_CHUNK_TOKENS, cleanup=True,
):
    if hf_datasets is None:
        hf_datasets = []
    os.makedirs(output_dir, exist_ok=True)

    # chunk는 Drive 영구 저장, 조립은 로컬 SSD
    chunks_dir = resolve_chunks_dir(output_dir, chunks_dir)
    assembly = resolve_assembly_scratch(output_dir, assembly_scratch)

    # 디스크 사전 확인 (조립 단계에서 train.npy + val.npy 동시 열기 → 로컬)
    if target_tokens and not smoke_test:
        assembly_need = int(target_tokens * 4 * 1.05)  # train+val ≈ target_tokens × 4 bytes
        check_disk_space(assembly, assembly_need)
        # Drive는 chunks + 최종 .npy 각각 한 벌씩 필요
        drive_need = int(target_tokens * 4 * 2.1)
        check_disk_space(output_dir, drive_need)

    print("=" * 60)
    print("📦 Chatbot Dataset Preparation (v8.2: persistent chunks)")
    print(f"   target_tokens       : {target_tokens:,}" if target_tokens else "   target_tokens       : unlimited")
    print(f"   chunk_size_tokens   : {chunk_size_tokens:,}")
    print(f"   chunks_dir (Drive)  : {chunks_dir}")
    print(f"   assembly (로컬 SSD) : {assembly}")
    print("=" * 60)

    print(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"   vocab_size={len(tokenizer)}")

    cfg_hash = config_hash(hf_datasets, block_size, target_tokens, chunk_size_tokens)
    progress = load_progress(chunks_dir)
    if progress is None or progress.get("config_hash") != cfg_hash:
        if progress is not None:
            print("⚠️ 설정 변경 감지 → 기존 chunk 폐기")
            cleanup_chunks_dir(chunks_dir)
        progress = {
            "config_hash": cfg_hash,
            "target_tokens": target_tokens,
            "total_tokens": 0,
            "datasets": {},
            "chunks": [],
        }
        save_progress(chunks_dir, progress)
    else:
        print(f"♻️ 진행 상태 복원: {len(progress['chunks'])} chunks, "
              f"{progress['total_tokens']:,} tokens 완료")

    if smoke_test:
        chunk_path = os.path.join(chunks_dir, "_chunk_0001.bin")
        with open(chunk_path, "wb") as f:
            dummy = [
                "<user>안녕하세요.<sep><assistant>네 안녕하세요.</s>",
                "<user>점심 추천?<sep><assistant>냉면이 좋습니다.</s>",
            ] * 40
            enc = tokenizer(dummy, add_special_tokens=False)["input_ids"]
            n = 0
            for ids in enc:
                if ids:
                    arr = np.asarray(ids, dtype=np.int32); arr.tofile(f); n += len(ids)
        progress["chunks"].append({"file": "_chunk_0001.bin", "tokens": n, "dataset": "smoke", "chunk_idx": 1})
        progress["total_tokens"] = n
        save_progress(chunks_dir, progress)
    else:
        for hf_entry in hf_datasets:
            if target_tokens and progress["total_tokens"] >= target_tokens:
                print(f"\n🎯 target_tokens 도달 ({progress['total_tokens']:,}).")
                break
            hf_name, _, hf_config = hf_entry.partition(":")
            hf_config = hf_config or None
            process_dataset_into_chunks(
                hf_name, hf_config, tokenizer, chunks_dir, progress,
                target_tokens, chunk_size_tokens, max_samples_per_dataset, batch_size,
            )

    print(f"\n📊 총 토큰: {progress['total_tokens']:,} ({len(progress['chunks'])} chunks)")
    if progress["total_tokens"] == 0:
        raise ValueError("No tokens collected.")

    print("\n📈 데이터셋별:")
    for ds_id, st in progress["datasets"].items():
        flag = "✅" if st["complete"] else "..."
        print(f"   {flag} {ds_id}: {st['tokens_emitted']:,} tokens ({st['samples_consumed']:,} samples)")

    assemble_chunks_into_npy(chunks_dir, assembly, output_dir, block_size, progress)

    # 조립 임시본은 항상 정리 (로컬 SSD, 곧 세션 끊겨도 어차피 소실)
    cleanup_assembly(assembly)

    if cleanup:
        print("\n🧹 chunk 정리 중 (--no_cleanup 시 보존)...")
        cleanup_chunks_dir(chunks_dir)
        try:
            os.rmdir(chunks_dir)
        except OSError:
            pass
    else:
        print(f"\n💾 chunk 보존: {chunks_dir}")

    print(f"\n📂 최종 경로: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--hf_datasets", nargs="*", default=[],
                        help="'name' 또는 'name:config'. 예: HuggingFaceFW/fineweb-2:kor_Hang")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/korean_output")
    parser.add_argument("--output_dir", type=str, default="train/chat_data")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--target_tokens", type=int, default=None)
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--chunk_size_tokens", type=int, default=DEFAULT_CHUNK_TOKENS,
                        help=f"chunk 1개당 최대 토큰 (기본 {DEFAULT_CHUNK_TOKENS:,} ≈ 2GB)")
    parser.add_argument("--chunks_dir", type=str, default=None,
                        help="chunk 영구 저장 위치. 미지정 시 {output_dir}/_chunks_persistent/")
    parser.add_argument("--assembly_scratch", type=str, default=None,
                        help="조립용 로컬 SSD. 미지정 시 Colab은 /content/_prepare_chat_assembly")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="작업 완료 후 chunk 보존 (재시도/디버깅용)")
    args = parser.parse_args()

    prepare_chat_data(
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        block_size=args.block_size,
        smoke_test=args.smoke_test,
        hf_datasets=args.hf_datasets,
        target_tokens=args.target_tokens,
        max_samples_per_dataset=args.max_samples_per_dataset,
        batch_size=args.batch_size,
        chunks_dir=args.chunks_dir,
        assembly_scratch=args.assembly_scratch,
        chunk_size_tokens=args.chunk_size_tokens,
        cleanup=not args.no_cleanup,
    )

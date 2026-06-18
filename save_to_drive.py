import os
import shutil

SRC_DIR = "/content/_prepare_chat_assembly"
DST_DIR = "/content/drive/MyDrive/korean_chat/chat_data"

os.makedirs(DST_DIR, exist_ok=True)

for fname in ["train.npy", "val.npy"]:
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DST_DIR, fname)
    if not os.path.exists(src):
        print(f"❌ 소스 없음: {src}")
        continue
    shutil.copy2(src, dst)
    size_gb = os.path.getsize(dst) / 1e9
    print(f"✅ {dst} ({size_gb:.2f} GB)")

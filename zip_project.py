import os
import zipfile


def zip_project(output_filename="transformer.zip"):
    """
    Colab 전송을 위해 핵심 소스코드 및 설정 파일들을 zip 파일로 압축합니다.
    __pycache__, 로컬 테스트 체크포인트, 임시 파일들은 자동으로 제외됩니다.
    """
    # 압축에 포함할 핵심 폴더 및 파일 리스트
    include_paths = [
        "model",
        "tokenizer",
        "train",
        "requirements-colab.txt",
        "requirements-local.txt",
        "Transformer_Model.ipynb",
    ]

    # 제외할 폴더 및 파일 명칭
    exclude_dirs = {
        "__pycache__",
        ".ipynb_checkpoints",
        "test_project",
        "test_project_wandb",
        "chat_data",
        "logs",
        "checkpoints",
    }
    exclude_files = {".DS_Store"}
    exclude_exts = {".pyc", ".pyo"}

    print(f"📦 Packaging project files into {output_filename}...")

    count = 0
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if not os.path.exists(path):
                print(f"⚠️ Warning: '{path}' not found. Skipping.")
                continue

            if os.path.isfile(path):
                zipf.write(path, path)
                print(f"  + Added file: {path}")
                count += 1
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    # __pycache__ 및 테스트 생성 폴더 등 필터링
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]

                    for file in files:
                        if file in exclude_files:
                            continue
                        if any(file.endswith(ext) for ext in exclude_exts):
                            continue

                        file_path = os.path.join(root, file)
                        # zip 내부 경로 그대로 등록 (예: model/ffn.py)
                        zipf.write(file_path, file_path)
                        count += 1

    print(f"✅ Success! Packaged {count} files into {output_filename}")


if __name__ == "__main__":
    zip_project()

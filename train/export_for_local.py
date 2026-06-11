import os
import argparse
import torch

def export_checkpoint(src_path, dst_path):
    """
    Colab(CUDA/BF16/DDP) 환경에서 학습된 체크포인트를
    로컬(MPS/CPU/FP32/Single Device) 환경에서 직접 로드하여 사용할 수 있도록 변환합니다.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source checkpoint not found at: {src_path}")
        
    print(f"Loading checkpoint from: {src_path}")
    # weights_only=False for backward compatibility with custom state structures
    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)

    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'")

    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {}
    
    for k, v in state_dict.items():
        # DDP(Distributed Data Parallel)로 학습 시 모델 파라미터명 앞에 붙는 'module.' 접두사를 제거
        key = k.replace("module.", "")
        # BF16 텐서를 FP32로 변환하여 CPU/MPS 로컬 추론 시 호환성 확보
        clean_state_dict[key] = v.float()

    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': clean_state_dict,
        'step': checkpoint.get('step', -1),
    }, dst_path)

    src_size = os.path.getsize(src_path) / 1e6
    dst_size = os.path.getsize(dst_path) / 1e6
    print(f"✅ Conversion complete: {src_size:.1f}MB (BF16) -> {dst_size:.1f}MB (FP32)")
    print(f"Saved checkpoint to: {dst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Colab에서 가져온 원본 체크포인트 파일 경로")
    parser.add_argument("--dst", type=str, required=True, help="로컬용으로 저장할 변환 체크포인트 파일 경로")
    args = parser.parse_args()
    
    export_checkpoint(args.src, args.dst)

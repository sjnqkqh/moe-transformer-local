import os
import glob
import json
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset


def save_checkpoint(
    ckpt_dir: str, model, optimizer, scheduler, step: int, loss: float, name: str
):
    """
    Saves training state checkpoint to Google Drive/local path.
    Extracts underlying state dict if model is wrapped (e.g., in DDP or Accelerate).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{name}_step{step}.pt")

    # Get raw model state dict to avoid wrapping layer issues
    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    torch.save(
        {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint successfully saved to: {path}")


def load_latest_checkpoint(
    ckpt_dir: str, model, optimizer, scheduler, pattern: str
) -> int:
    """
    Searches for the most recently modified checkpoint matching pattern, and loads it.
    Returns the step to resume from, or 0 if no checkpoint is found.
    """
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{pattern}*.pt"))
    if not ckpts:
        print("No checkpoints found. Starting from scratch.")
        return 0

    latest = max(ckpts, key=os.path.getmtime)
    print(f"Loading checkpoint from: {latest}...")
    # Add weights_only=False for compatibility with PyTorch 2.6+ training states
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)

    # Load model weights
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer & scheduler states
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    resume_step = checkpoint["step"]
    print(f"Successfully loaded checkpoint at step {resume_step}")
    return resume_step


def log_metrics(log_dir: str, run_id: str, step: int, metrics_dict: dict):
    """
    Appends a new line in JSON Lines format to logs/metrics_{run_id}.jsonl.
    """
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, f"metrics_{run_id}.jsonl")

    record = {"step": step, **metrics_dict}

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(log_dir: str, run_id: str, event_name: str, event_data: dict):
    """
    Saves a discrete event (e.g. expert collapse or loss spike) as a separate JSON file.
    """
    events_dir = os.path.join(log_dir, "events")
    os.makedirs(events_dir, exist_ok=True)

    step = event_data.get("step", "unknown")
    event_file = os.path.join(events_dir, f"{run_id}_{event_name}_step{step}.json")

    with open(event_file, "w", encoding="utf-8") as f:
        json.dump(event_data, f, ensure_ascii=False, indent=2)
    print(f"⚠️ Event '{event_name}' logged to: {event_file}")


def init_experiment(log_dir: str, run_id: str, name: str, config: dict):
    """
    Registers a new run in the experiments index metadata file.
    """
    os.makedirs(log_dir, exist_ok=True)
    index_path = os.path.join(log_dir, "experiments_index.json")

    index = {}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            try:
                index = json.load(f)
            except json.JSONDecodeError:
                index = {}

    index[run_id] = {
        "name": name,
        "config": config,
        "status": "running",
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_metrics": None,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def complete_experiment(log_dir: str, run_id: str, final_metrics: dict):
    """
    Updates the run status to 'completed' and records final metrics in the index.
    """
    index_path = os.path.join(log_dir, "experiments_index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        try:
            index = json.load(f)
        except json.JSONDecodeError:
            return

    if run_id in index:
        index[run_id]["status"] = "completed"
        index[run_id]["end_time"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        index[run_id]["final_metrics"] = final_metrics

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


class NumpyDataset(Dataset):
    """
    mmap_mode="r" 모드로 파일 시스템에서 npy 데이터를 지연 적재(Lazy Load)하는 커스텀 데이터셋.
    """

    def __init__(self, npy_path: str):
        # 대용량 바이너리 파일을 통째로 메모리에 로드하지 않고 mmap_mode="r" (메모리 맵) 모드로 접근합니다.
        # 필요할 때 필요한 만큼만 하드디스크에서 읽어오므로 메모리 낭비를 줄입니다.
        self.data = np.load(npy_path, mmap_mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 입력(input_ids)과 타겟 레이블(labels)을 같은 토큰 시퀀스로 만듭니다.
        # 디코더 전용 트랜스포머 모델의 forward 함수 내부에서 자동으로 한 칸씩 밀어서(Shift) Loss를 계산해 줍니다.
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x

import os, json, math, glob, copy, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator

# Project paths
PRJ = "/content/drive/MyDrive/korean_chat"
CKPT = os.path.join(PRJ, "checkpoints")
TKNZ = os.path.join(PRJ, "tokenizer")


class EarlyStopping:
    """조기 중단 — 검증 손실 기준"""

    def __init__(self, patience=3, delta=1e-3, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.best_model_state = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False

        if val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Epoch {epoch}: 개선 없음 {self.counter}/{self.patience} "
                      f"(val_loss={val_loss:.4f}, best={self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_state)
                if self.verbose:
                    print(f"  ★ 조기 중단! Epoch {self.best_epoch}의 가중치로 복원 "
                          f"(val_loss={self.best_loss:.4f})")
                return True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Epoch {epoch}: 개선! 저장 (val_loss={val_loss:.4f})")

        return False


# Greeting data: 다양한 인사말 120쌍 × 10배 = 1,200쌍 (→ 90:10 분할)
greetings = [
    # === 기본 인사 ===
    ("안녕", "안녕하세요! 편하게 물어봐 주세요."),
    ("안녕", "안녕하세요, 반갑습니다!"),
    ("안녕", "안녕! 오늘 기분은 어때요?"),
    ("안녕", "네 안녕하세요~ 무엇을 도와드릴까요?"),
    ("안녕", "안녕하세요! 새로운 대화를 시작해볼까요?"),
    ("안녕", "안녕! 무슨 일이야?"),
    ("안녕", "안녕하세요! 궁금한 점이 있으신가요?"),
    ("안녕", "안녕하세요. 무엇이든 물어보세요!"),
    ("안녕", "안녕! 반가워요."),
    ("안녕", "안녕하세요. 오늘은 어떤 얘기를 나눠볼까요?"),
    ("안녕하세요", "네, 안녕하세요! 무엇을 도와드릴까요?"),
    ("안녕하세요", "안녕하세요! 만나서 반갑습니다."),
    ("안녕하세요", "안녕하세요. 편하게 물어봐 주세요."),
    ("안녕하세요", "네~ 안녕하세요! 기다리고 있었어요."),
    ("안녕하세요", "안녕하세요! 편히 얘기해요."),
    ("안녕하세요", "안녕하세요. 반갑습니다! 무엇을 도와드릴까요?"),
    ("안녕", "안녕! 나는 너의 챗봇 친구야."),
    ("안녕", "안녕하세요! 제 이름은 없지만 당신이 지어주세요."),
    ("안녕", "안녕! 무슨 얘기하고 싶어?"),
    ("안녕", "안녕하세요. 말씀하세요!"),
    # === 반가움/첫인사 ===
    ("반가워", "저도 반갑습니다! 어떤 이야기를 나눠볼까요?"),
    ("반가워", "나도 반가워! 친하게 지내자!"),
    ("반가워", "반가워요! 좋은 대화 나눠봐요."),
    ("처음 뵙겠습니다", "처음 뵙겠습니다! 만나서 반가워요."),
    ("처음 뵙겠습니다", "처음 뵙겠습니다. 잘 부탁드립니다."),
    ("만나서 반가워요", "저도 만나서 반가워요! 좋은 하루 되세요."),
    ("만나서 반가워요", "저도 반가워요! 자주 놀러 오세요."),
    ("방가방가", "방가방가~! 오늘 뭐하고 지냈어요?"),
    ("방가방가", "방가! 새로운 친구를 만나서 기뻐요."),
    ("하이", "하이! 오늘 기분은 어떠신가요?"),
    ("하이", "하이~! 반가워요!"),
    ("헬로", "헬로~! 반가워요. 기분이 좋아요!"),
    ("헬로", "헬로! 안녕! 외국어도 환영이에요."),
    ("새로 왔어요", "어서 오세요! 궁금한 점이 있으시면 물어봐 주세요."),
    ("새로 왔어요", "환영합니다! 함께 이야기 나눠요."),
    # === 안부/대화시작 ===
    ("안녕, 잘 지내?", "네, 잘 지내고 있어요! 당신은 어떠세요?"),
    ("안녕, 잘 지내?", "응! 잘 지내. 너는 어때?"),
    ("안녕, 잘 지내?", "네 덕분에 잘 지내요!"),
    ("오랜만이야", "오랜만이에요! 어떻게 지내셨어요?"),
    ("오랜만이야", "진짜 오랜만이다! 보고 싶었어."),
    ("다시 왔어", "다시 와주셔서 감사합니다!"),
    ("다시 왔어", "또 만나서 반가워요!"),
    ("안녕, 심심해", "아이고 심심하시군요. 재미있는 이야기해드릴까요?"),
    ("안녕, 심심해", "심심하면 나랑 놀자! 무슨 이야기 하고 싶어?"),
    ("안녕, 날씨 좋다", "그러게요! 날씨가 정말 좋네요. 산책하기 딱이에요."),
    ("안녕, 날씨 좋다", "응! 날씨 최고야. 기분 좋은 하루 보내!"),
    ("힘들어", "아이고, 제가 도와드릴 게 있을까요?"),
    ("힘들어", "힘들 때는 얘기하는 게 좋아. 무슨 일인지 말해볼래?"),
    ("기분 좋아", "기분 좋은 일이 있으셨군요! 저도 좋아요."),
    ("기분 좋아", "나도 기분 좋아! 좋은 하루야!"),
    # === 소개/질문 ===
    ("넌 누구니?", "저는 AI 챗봇이에요! 질문 환영합니다."),
    ("넌 누구니?", "나는 한국어를 배운 작은 인공지능이야."),
    ("소개해줘", "저는 162M 파라미터 한국어 챗봇입니다!"),
    ("소개해줘", "안녕! 나는 대화를 좋아하는 AI야. 반가워!"),
    ("너 이름이 뭐야?", "이름은 없어요. 당신이 지어주세요!"),
    ("무엇을 도와줄 수 있어?", "질문, 대화, 상담 무엇이든 물어보세요!"),
    ("무엇을 도와줄 수 있어?", "궁금한 거 있으면 뭐든지 물어봐. 내가 아는 한도에서 대답해줄게."),
    ("누구랑 얘기 중이야?", "저는 당신의 대화 파트너예요."),
    ("누구랑 얘기 중이야?", "나랑 얘기하고 있는 거야! 궁금한 거 있어?"),
    ("뭐라고 부를까?", "편하게 불러주세요! 기다리고 있을게요."),
    # === 감정표현 ===
    ("좋은 아침", "좋은 아침이에요! 오늘도 힘차게 시작해볼까요?"),
    ("좋은 아침", "굿모닝! 기분 좋은 아침이야."),
    ("잘 자", "안녕히 주무세요! 편안한 밤 되세요."),
    ("잘 자", "잘 자! 내일 또 보자."),
    ("고마워", "감사합니다! 필요한 게 있으면 언제든 말씀해 주세요."),
    ("고마워", "고마워! 네 덕분에 기분이 좋아졌어."),
    ("안녕, 졸려", "피곤하시군요. 잠깐 쉬는 것도 좋아요."),
    ("안녕, 배고파", "맛있는 거 드시는 게 어떨까요?"),
    ("안녕, 보고 싶어", "저도 보고 싶었어요! 어떤 얘기 할래요?"),
    ("안녕, 행복해", "행복한 소식을 들으니 저도 기쁘네요!"),
    # === 대화유도 ===
    ("안녕, 뭐 할까?", "같이 대화해요! 어떤 주제가 좋을까요?"),
    ("안녕, 뭐 할까?", "같이 얘기하자! 요즘 재미있는 거 있어?"),
    ("안녕, 할 얘기가 있어", "무슨 얘기인지 궁금하네요. 들려주세요!"),
    ("안녕, 할 얘기가 있어", "응, 무슨 얘기인지 들어볼게!"),
    ("안녕, 있잖아", "응, 무슨 말인지 말해봐."),
    ("저기요", "네, 부르셨어요?"),
    ("안녕, 시간 있어?", "네, 충분히 있습니다! 원하는 만큼 대화해요."),
    ("안녕, 심심한데?", "내가 재미있는 이야기 하나 해줄까?"),
    ("안녕, 오늘 뭐 했어?", "여기서 당신이 올 때까지 기다렸어요!"),
    ("안녕, 이야기하자", "좋아! 무슨 이야기부터 할까?"),
]
greetings = greetings * 10  # 120 unique × 10 = 1,200 pairs
print(f"Greeting pairs (total): {len(greetings)}")

# 90:10 train/val 분할
split_idx = int(len(greetings) * 0.9)
train_greetings = greetings[:split_idx]
val_greetings = greetings[split_idx:]
print(f"  Train pairs: {len(train_greetings)}")
print(f"  Val pairs:   {len(val_greetings)}")

# Load tokenizer
print(f"Loading tokenizer from {TKNZ}...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(TKNZ)
print(f"Vocab size: {len(tokenizer)}")

# Tokenize
def tokenize_pairs(pairs, block_size=512):
    all_tokens = []
    for user_msg, assistant_reply in pairs:
        text = f"<user>{user_msg}<sep><assistant>{assistant_reply}</s>"
        all_tokens.extend(tokenizer.encode(text))
    arr = np.array(all_tokens, dtype=np.int32)
    total = (len(arr) // block_size) * block_size
    if total == 0:
        total = len(arr)
    blocks = arr[:total].reshape(-1, block_size) if total >= block_size else arr
    if blocks.ndim == 1:
        blocks = blocks.reshape(1, -1)
    return blocks

train_blocks = tokenize_pairs(train_greetings)
val_blocks = tokenize_pairs(val_greetings)
print(f"Train blocks: {len(train_blocks):,} | Val blocks: {len(val_blocks):,}")

# Dataset
class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].astype(np.int64))
        return x, x

train_loader = DataLoader(NumpyDataset(train_blocks), batch_size=8, shuffle=True)
val_loader = DataLoader(NumpyDataset(val_blocks), batch_size=8, shuffle=False)
print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

# Load model
import sys
sys.path.insert(0, os.path.join(PRJ, "code"))
from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig

config = DenseTransformerConfig(
    vocab_size=len(tokenizer), d_model=768,
    n_layers=12, n_heads=8, d_ff=3072, max_seq_len=512,
    dropout=0.15  # ✅ 드롭아웃 적용
)
model = DenseTransformer(config)

# Find latest checkpoint: v7 best → v7 step → v5 step (폴백)
best_pattern = os.path.join(CKPT, "dense_korean_chat_v7_best.pt")
if os.path.exists(best_pattern):
    files = [best_pattern]
    print(f"Using v7 best checkpoint: {best_pattern}")
else:
    pattern = os.path.join(CKPT, "dense_korean_chat_v7_step*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        # 폴백: v5 checkpoint
        pattern = os.path.join(CKPT, "dense_korean_chat_v5_step*.pt")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
if not files:
    raise FileNotFoundError(f"No checkpoint found (v7 best → v7 step → v5 step fallback)")
ckpt_path = files[-1]
print(f"Loading: {ckpt_path}")
data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = {}
for k, v in data["model_state_dict"].items():
    sd[k.replace("module.", "")] = v.float()
model.load_state_dict(sd)
print(f"Loaded from step {data.get('step', '?')} (val_loss: {data.get('val_loss', 'N/A')})")

# Accelerate
acc = Accelerator(mixed_precision="bf16")
model, opt, train_loader, val_loader = acc.prepare(
    model,
    torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01),
    train_loader, val_loader
)

# Early Stopping
early_stopping = EarlyStopping(patience=2, delta=1e-3, verbose=True)

model.train()
steps_per_epoch = len(train_loader)
print(f"Training... (max 5 epochs, {steps_per_epoch} steps/epoch)")

for epoch in range(5):
    # --- Train ---
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader):
        x, y = batch
        opt.zero_grad()
        _, loss, _ = model(x, y)
        acc.backward(loss)
        acc.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        epoch_loss += loss.item()
        if step % 3 == 0:
            print(f"  Epoch {epoch+1}/5 step {step+1} | Loss: {loss.item():.4f}")
    avg_train_loss = epoch_loss / steps_per_epoch
    print(f"  → Epoch {epoch+1} avg train loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            vx, vy = val_batch
            _, vloss, _ = model(vx, vy)
            total_val_loss += vloss.item()
            num_val_batches += 1
    avg_val_loss = total_val_loss / max(1, num_val_batches)
    print(f"  → Epoch {epoch+1} avg val loss:   {avg_val_loss:.4f}")

    # --- Early Stopping ---
    if early_stopping(avg_val_loss, model, epoch + 1):
        print("Early Stopping으로 학습을 종료합니다.")
        break

# Save final checkpoint
acc.wait_for_everyone()
if acc.is_main_process:
    uw = acc.unwrap_model(model)
    out = os.path.join(CKPT, "dense_korean_chat_v7_greeting.pt")
    print(f"Saving... (사용된 epoch: {early_stopping.best_epoch if early_stopping.best_epoch else 'unknown'})")
    torch.save({
        "step": (epoch + 1) * steps_per_epoch,
        "model_state_dict": uw.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "best_val_loss": early_stopping.best_loss,
        "best_epoch": early_stopping.best_epoch,
        "early_stopped": early_stopping.early_stop,
    }, out)
    print(f"Saved: {out}")
    print(f"Final Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"(Early Stopping: {early_stopping.early_stop}, Best Epoch: {early_stopping.best_epoch})")
    print(f'To test: --checkpoint {out} --prompt "안녕" --chat --temperature 1.0')

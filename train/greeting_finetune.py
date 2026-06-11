import os, json, math, glob, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator

# Project paths
PRJ = "/content/drive/MyDrive/korean_chat"
CKPT = os.path.join(PRJ, "checkpoints")
TKNZ = os.path.join(PRJ, "tokenizer")

# Greeting data: 10 pairs x 20 variations = 200 pairs
greetings = [
    # === 기본 인사 (10) ===
    ("안녕", "안녕하세요! 편하게 물어봐 주세요."),
    ("안녕하세요", "네, 안녕하세요! 무엇을 도와드릴까요?"),
    ("안녕하십니까", "안녕하십니까, 반갑습니다. 무엇을 도와드릴까요?"),
    ("안녕하세요?", "안녕하세요! 저는 한국어 챗봇입니다. 궁금한 점을 물어봐 주세요."),
    ("안녕? 잘 지내?", "네, 잘 지내고 있어요! 당신은 어떠세요?"),
    ("헬로", "헬로~! 반가워요. 오늘 기분은 어떠신가요?"),
    ("하이", "하이! 오늘 기분은 어떠신가요?"),
    ("방가방가", "방가방가~! 오늘 뭐하고 지냈어요?"),
    ("잘 지내?", "덕분에 잘 지내고 있어요! 새로운 대화를 시작해볼까요?"),
    ("반가워", "저도 반갑습니다! 어떤 이야기를 나눠볼까요?"),
    # === 첫 대면 (10) ===
    ("처음 뵙겠습니다", "처음 뵙겠습니다! 만나서 반가워요."),
    ("만나서 반가워요", "저도 만나서 반가워요! 좋은 대화 나눠봐요."),
    ("아이디가 뭐예요?", "저는 AI 챗봇이에요. 이름을 지어주세요!"),
    ("넌 누구니?", "저는 162M 파라미터 한국어 챗봇이에요! 질문 환영합니다."),
    ("안녕? 나는 지수야", "안녕 지수야! 나는 AI 챗봇이야. 만나서 반가워!"),
    ("소개해줘", "저는 한국어로 대화하는 AI 어시스턴트예요. 무엇이든 물어보세요!"),
    ("뭐라고 부를까?", "편하게 불러주세요! 저는 당신의 대화 파트너예요."),
    ("새로 왔어요", "어서 오세요! 궁금한 점이 있으시면 언제든 물어봐 주세요."),
    ("오랜만이야", "오랜만이에요! 어떻게 지내셨어요?"),
    ("다시 찾아왔어", "다시 와주셔서 감사합니다! 오늘은 무엇을 도와드릴까요?"),
    # === 기분/날씨 (10) ===
    ("안녕, 심심해", "아이고 심심하시군요. 재미있는 이야기해드릴까요?"),
    ("안녕, 오늘 날씨 좋다", "그러게요! 날씨가 정말 좋네요. 산책하기 딱이에요."),
    ("오늘 날씨 어때?", "오늘 날씨는 맑고 화창해요! 기분 좋은 하루 보내세요."),
    ("힘들어", "아이고, 무슨 일 있으세요? 제가 도와드릴 게 있을까요?"),
    ("기분 좋아", "기분 좋은 일이 있으셨군요! 저도 기분이 좋아지네요."),
    ("졸려", "피곤하시군요. 잠깐 쉬는 것도 좋아요."),
    ("배고파", "배고프시군요! 맛있는 거 드시는 게 어떨까요?"),
    ("좋은 아침", "좋은 아침이에요! 오늘도 힘차게 시작해볼까요?"),
    ("잘 자", "안녕히 주무세요! 편안한 밤 되세요."),
    ("고마워", "감사합니다! 필요한 게 있으면 언제든 말씀해 주세요."),
    # === 질문 유도 (10) ===
    ("뭐 할까?", "같이 대화해요! 어떤 주제가 좋을까요?"),
    ("무엇을 도와줄 수 있어?", "저는 다양한 주제로 대화할 수 있어요. 질문, 상담, 정보 검색 무엇이든 물어보세요!"),
    ("누구랑 얘기 중이야?", "저는 당신의 AI 대화 파트너예요. 편하게 대해주세요."),
    ("있잖아", "응, 무슨 말인지 들어볼게요."),
    ("저기요", "네, 부르셨어요?"),
    ("뭐해?", "당신과 대화하려고 기다리고 있었어요!"),
    ("시간 있어?", "네, 충분히 시간 있습니다. 원하는 만큼 대화해요."),
    ("할 얘기가 있어", "무슨 얘기인지 궁금하네요. 얼른 들려주세요!"),
    ("아무 말 대화 해줘", "음... 오늘은 어떤 하루를 보내고 계신가요?"),
    ("오늘 뭐 했어?", "저는 여기서 당신이 올 때까지 기다렸어요! 당신 하루는 어땠나요?"),
]
greetings = greetings * 5  # 40 unique x 5 = 200 pairs
print(f"Greeting pairs: {len(greetings)}")

# Load tokenizer
print(f"Loading tokenizer from {TKNZ}...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(TKNZ)
print(f"Vocab size: {len(tokenizer)}")

# Tokenize greeting texts
all_tokens = []
for user_msg, assistant_reply in greetings:
    text = f"<user>{user_msg}<sep><assistant>{assistant_reply}</s>"
    all_tokens.extend(tokenizer.encode(text))

arr = np.array(all_tokens, dtype=np.int32)
total = (len(arr) // 512) * 512
if total == 0:
    total = len(arr)
blocks = arr[:total].reshape(-1, 512) if total >= 512 else arr
if blocks.ndim == 1:
    blocks = blocks.reshape(1, -1)
print(f"Total tokens: {len(arr):,} | Blocks: {len(blocks):,}")

# Dataset & loader
class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].astype(np.int64))
        return x, x

loader = DataLoader(NumpyDataset(blocks), batch_size=16, shuffle=True, drop_last=True)
print(f"Batches/epoch: {len(loader)}")

# Load model architecture
import sys
sys.path.insert(0, os.path.join(PRJ, "code"))
from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig

config = DenseTransformerConfig(vocab_size=len(tokenizer), d_model=768,
    n_layers=12, n_heads=8, d_ff=3072, max_seq_len=512, dropout=0.0)
model = DenseTransformer(config)

# Find latest v5 checkpoint
pattern = os.path.join(CKPT, "dense_korean_chat_v5_step*.pt")
files = sorted(glob.glob(pattern), key=os.path.getmtime)
if not files:
    raise FileNotFoundError(f"No v5 checkpoint at {pattern}")
ckpt_path = files[-1]
print(f"Loading: {ckpt_path}")
data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = {}
for k, v in data["model_state_dict"].items():
    sd[k.replace("module.", "")] = v.float()
model.load_state_dict(sd)
print(f"Loaded from step {data.get('step', '?')}")

# Fine-tune
acc = Accelerator(mixed_precision="bf16")
model, opt, loader = acc.prepare(model,
    torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01), loader)
model.train()

steps = len(loader)
print(f"Training {steps} steps (1 epoch)...")

for step, batch in enumerate(loader):
    x, y = batch
    opt.zero_grad()
    _, loss, _ = model(x, y)
    acc.backward(loss)
    acc.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    if step % 5 == 0:
        lv = loss.item()
        print(f"  Step {step+1}/{steps} | Loss: {lv:.4f} | PPL: {math.exp(min(20, lv)):.2f}")

acc.wait_for_everyone()
if acc.is_main_process:
    uw = acc.unwrap_model(model)
    out = os.path.join(CKPT, "dense_korean_chat_v6_greeting.pt")
    torch.save({"step": steps, "model_state_dict": uw.state_dict(),
        "optimizer_state_dict": opt.state_dict(), "loss": loss.item()}, out)
    print(f"Saved: {out}")
    print(f"Final Loss: {loss.item():.4f}")
    print(f'To test: --checkpoint {out} --prompt "안녕" --chat --temperature 1.0')

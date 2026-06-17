import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 프로젝트 루트 경로를 sys.path에 추가하여 패키지 임포트 지원
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig
from transformers import PreTrainedTokenizerFast
from train.generate import chat_generate

# 전역 상태 변수 정의
model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 기동 시 모델과 토크나이저를 로드하고, 종료 시 메모리를 정리합니다.
    """
    global model, tokenizer, device

    # 디바이스 설정 (MPS -> CUDA -> CPU 순서로 탐색)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"🚀 FastAPI starting up. Using device: {device}")

    # 토크나이저 로드
    tokenizer_path = "tokenizer/korean_output"
    if not os.path.exists(tokenizer_path):
        # 만약 로컬에 korean_output이 없으면 output을 폴백으로 체크
        tokenizer_path = "tokenizer/output"

    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer not found. Run train_tokenizer.py first.")

    # 모델 설정 및 인스턴스 생성
    vocab_size = len(tokenizer)
    config = DenseTransformerConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.0,
    )
    model = DenseTransformer(config)

    # 체크포인트 로드
    model_path = "serve/model/korean_chat_v7_greet_tuning.pt"
    if not os.path.exists(model_path):
        print(f"⚠️ Checkpoint not found at {model_path}. Trying smoke test model...")
        model_path = "test_project/checkpoints/dense_chat_smoke_step10.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found. Train the model first."
            )

    print(f"Loading model checkpoint from {model_path}...")
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        # module. 접두사 제거 로직 (DDP 대응)
        state_dict = ckpt["model_state_dict"]
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            clean_state_dict[key] = v.float()

        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()
        print("✅ Model loaded and ready for inference.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e

    yield  # 애플리케이션 가동

    # 종료 시 메모리 정리
    print("Stopping FastAPI application and releasing resources.")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# FastAPI 앱 생성
app = FastAPI(
    title="Korean Chatbot Server",
    description="Dense Transformer 기반의 한국어 챗봇 서비스 API",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str
    temperature: float = 1.0
    max_new_tokens: int = 100
    top_k: int = 30
    top_p: float = 0.95
    repetition_penalty: float = 1.5


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    유저 메시지를 받아 챗봇 답변을 생성하여 반환합니다.
    """
    global model, tokenizer, device
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        reply = chat_generate(
            model=model,
            tokenizer=tokenizer,
            user_message=req.message,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            device=device,
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    웹 브라우저에서 직접 테스트할 수 있는 대화형 인터페이스를 렌더링합니다.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Korean Dense Transformer</title>
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            :root {
                --blue: #0066cc;
                --blue-dark: #0071e3;
                --ink: #1d1d1f;
                --ink-muted: #6e6e73;
                --canvas: #ffffff;
                --parchment: #f5f5f7;
                --hairline: #d2d2d7;
                --on-dark: #ffffff;
                --surface-black: #000000;
                --font-display: "SF Pro Display", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                --font-text: "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            }

            body {
                font-family: var(--font-text);
                background: var(--parchment);
                color: var(--ink);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            /* Global nav */
            .global-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 44px;
                background: var(--surface-black);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 100;
            }

            .global-nav-inner {
                width: 100%;
                max-width: 980px;
                padding: 0 22px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .nav-logo {
                color: var(--on-dark);
                font-size: 17px;
                font-weight: 400;
                letter-spacing: -0.374px;
                text-decoration: none;
            }

            .nav-status {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                font-weight: 400;
                letter-spacing: -0.12px;
                color: rgba(255,255,255,0.56);
            }

            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #30d158;
                flex-shrink: 0;
            }

            /* Page layout */
            .page-body {
                margin-top: 44px;
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 48px 22px 32px;
            }

            /* Sub-nav / title strip */
            .sub-nav {
                width: 100%;
                max-width: 740px;
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                margin-bottom: 24px;
            }

            .sub-nav-title {
                font-family: var(--font-display);
                font-size: 21px;
                font-weight: 600;
                letter-spacing: 0.231px;
                color: var(--ink);
            }

            .sub-nav-model {
                font-size: 12px;
                font-weight: 400;
                letter-spacing: -0.12px;
                color: var(--ink-muted);
            }

            /* Chat card */
            .chat-card {
                width: 100%;
                max-width: 740px;
                background: var(--canvas);
                border: 1px solid var(--hairline);
                border-radius: 18px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                flex: 1;
                min-height: 0;
                max-height: calc(100vh - 44px - 48px - 32px - 56px);
            }

            /* Messages */
            .chat-messages {
                flex: 1;
                padding: 32px 32px 24px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 24px;
                scroll-behavior: smooth;
            }

            .chat-messages::-webkit-scrollbar {
                width: 4px;
            }
            .chat-messages::-webkit-scrollbar-track {
                background: transparent;
            }
            .chat-messages::-webkit-scrollbar-thumb {
                background: var(--hairline);
                border-radius: 9999px;
            }

            .message {
                max-width: 72%;
                display: flex;
                flex-direction: column;
                gap: 5px;
                animation: msgIn 0.22s ease-out both;
            }

            @keyframes msgIn {
                from { opacity: 0; transform: translateY(8px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            .message.user { align-self: flex-end; }
            .message.bot  { align-self: flex-start; }

            .message-sender {
                font-size: 12px;
                font-weight: 400;
                letter-spacing: -0.12px;
                color: var(--ink-muted);
                padding: 0 5px;
            }

            .message.user .message-sender { text-align: right; }

            .message-bubble {
                padding: 11px 17px;
                border-radius: 18px;
                font-size: 17px;
                font-weight: 400;
                line-height: 1.47;
                letter-spacing: -0.374px;
            }

            .message.user .message-bubble {
                background: var(--blue);
                color: var(--on-dark);
                border-bottom-right-radius: 5px;
            }

            .message.bot .message-bubble {
                background: var(--parchment);
                color: var(--ink);
                border: 1px solid var(--hairline);
                border-bottom-left-radius: 5px;
            }

            /* Typing indicator */
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 2px 4px;
            }

            .typing-dot {
                width: 6px;
                height: 6px;
                background: var(--ink-muted);
                border-radius: 50%;
                opacity: 0.4;
                animation: bounce 1.4s infinite ease-in-out;
            }
            .typing-dot:nth-child(1) { animation-delay: 0s; }
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }

            @keyframes bounce {
                0%, 100% { transform: translateY(0); opacity: 0.4; }
                50%       { transform: translateY(-4px); opacity: 0.9; }
            }

            /* Parameter strip */
            .param-strip {
                display: flex;
                gap: 8px;
                padding: 10px 32px;
                border-top: 1px solid var(--hairline);
                background: var(--parchment);
                flex-wrap: wrap;
            }

            .param-chip {
                display: flex;
                align-items: center;
                gap: 5px;
                background: var(--canvas);
                border: 1px solid var(--hairline);
                border-radius: 9999px;
                padding: 4px 12px;
                font-size: 12px;
                font-weight: 400;
                letter-spacing: -0.12px;
                color: var(--ink-muted);
            }

            .param-chip label {
                white-space: nowrap;
            }

            .param-chip input {
                background: transparent;
                border: none;
                outline: none;
                font-size: 12px;
                font-weight: 600;
                color: var(--blue);
                width: 38px;
                font-family: var(--font-text);
                letter-spacing: -0.12px;
                padding: 0;
            }

            /* Input area */
            .chat-input-area {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 12px 20px;
                border-top: 1px solid var(--hairline);
                background: var(--canvas);
            }

            .chat-input {
                flex: 1;
                height: 44px;
                background: var(--parchment);
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 9999px;
                padding: 0 20px;
                font-family: var(--font-text);
                font-size: 17px;
                font-weight: 400;
                letter-spacing: -0.374px;
                color: var(--ink);
                outline: none;
                transition: border-color 0.15s, box-shadow 0.15s;
            }

            .chat-input::placeholder {
                color: var(--ink-muted);
            }

            .chat-input:focus {
                border-color: var(--blue-dark);
                box-shadow: 0 0 0 2px rgba(0, 113, 227, 0.18);
            }

            .send-btn {
                height: 44px;
                padding: 0 18px;
                background: var(--blue);
                border: none;
                border-radius: 9999px;
                color: var(--on-dark);
                font-family: var(--font-text);
                font-size: 17px;
                font-weight: 400;
                letter-spacing: -0.374px;
                cursor: pointer;
                transition: transform 0.1s;
                white-space: nowrap;
                display: flex;
                align-items: center;
            }

            .send-btn:active {
                transform: scale(0.95);
            }

            .send-btn:disabled {
                opacity: 0.44;
                cursor: default;
            }
        </style>
    </head>
    <body>
        <nav class="global-nav">
            <div class="global-nav-inner">
                <span class="nav-logo">Dense Transformer</span>
                <div class="nav-status">
                    <span class="status-dot"></span>
                    Model Online
                </div>
            </div>
        </nav>

        <main class="page-body">
            <div class="sub-nav">
                <span class="sub-nav-title">Korean Chatbot</span>
                <span class="sub-nav-model">162M · BF16</span>
            </div>

            <div class="chat-card">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot">
                        <span class="message-sender">Assistant</span>
                        <div class="message-bubble">
                            안녕하세요! 대화를 시작해보세요.
                        </div>
                    </div>
                </div>

                <div class="param-strip">
                    <div class="param-chip">
                        <label for="paramTemp">Temp</label>
                        <input type="number" step="0.1" min="0.1" max="1.5" id="paramTemp" value="1.0">
                    </div>
                    <div class="param-chip">
                        <label for="paramTopP">Top-P</label>
                        <input type="number" step="0.05" min="0.1" max="1.0" id="paramTopP" value="0.95">
                    </div>
                    <div class="param-chip">
                        <label for="paramRep">Rep Pen</label>
                        <input type="number" step="0.1" min="1.0" max="2.0" id="paramRep" value="1.5">
                    </div>
                    <div class="param-chip">
                        <label for="paramMaxTokens">Max Tokens</label>
                        <input type="number" step="10" min="10" max="256" id="paramMaxTokens" value="100">
                    </div>
                </div>

                <div class="chat-input-area">
                    <input type="text" class="chat-input" id="chatInput" placeholder="메시지를 입력하세요…" autofocus>
                    <button class="send-btn" id="sendBtn">보내기</button>
                </div>
            </div>
        </main>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const chatInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendBtn');
            const paramTemp = document.getElementById('paramTemp');
            const paramTopP = document.getElementById('paramTopP');
            const paramRep = document.getElementById('paramRep');
            const paramMaxTokens = document.getElementById('paramMaxTokens');

            function appendMessage(sender, text, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

                const senderSpan = document.createElement('span');
                senderSpan.className = 'message-sender';
                senderSpan.textContent = sender;

                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'message-bubble';
                bubbleDiv.textContent = text;

                messageDiv.appendChild(senderSpan);
                messageDiv.appendChild(bubbleDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function showTypingIndicator() {
                const indicatorDiv = document.createElement('div');
                indicatorDiv.className = 'message bot';
                indicatorDiv.id = 'typingIndicator';

                const senderSpan = document.createElement('span');
                senderSpan.className = 'message-sender';
                senderSpan.textContent = 'Assistant';

                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'message-bubble';

                const indicator = document.createElement('div');
                indicator.className = 'typing-indicator';
                for (let i = 0; i < 3; i++) {
                    const dot = document.createElement('span');
                    dot.className = 'typing-dot';
                    indicator.appendChild(dot);
                }

                bubbleDiv.appendChild(indicator);
                indicatorDiv.appendChild(senderSpan);
                indicatorDiv.appendChild(bubbleDiv);
                chatMessages.appendChild(indicatorDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function removeTypingIndicator() {
                const el = document.getElementById('typingIndicator');
                if (el) el.remove();
            }

            async function handleSend() {
                const text = chatInput.value.trim();
                if (!text) return;

                chatInput.value = '';
                sendBtn.disabled = true;
                appendMessage('나', text, true);
                showTypingIndicator();

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: text,
                            temperature: parseFloat(paramTemp.value) || 1.0,
                            top_p: parseFloat(paramTopP.value) || 0.95,
                            repetition_penalty: parseFloat(paramRep.value) || 1.5,
                            max_new_tokens: parseInt(paramMaxTokens.value) || 100
                        })
                    });

                    const data = await response.json();
                    removeTypingIndicator();

                    if (response.ok) {
                        appendMessage('Assistant', data.reply || '(응답 없음)');
                    } else {
                        appendMessage('Assistant', `오류: ${data.detail || '응답 생성 실패'}`);
                    }
                } catch (error) {
                    removeTypingIndicator();
                    appendMessage('Assistant', `오류: ${error.message}`);
                } finally {
                    sendBtn.disabled = false;
                    chatInput.focus();
                }
            }

            sendBtn.addEventListener('click', handleSend);
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') handleSend();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

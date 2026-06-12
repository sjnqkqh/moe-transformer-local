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
    model_path = "serve/model/korean_chat.pt"
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
    웹 브라우저에서 직접 테스트할 수 있는 세련된 대화형 인터페이스를 렌더링합니다.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transformer Korean Chatbot</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                --panel-bg: rgba(30, 41, 59, 0.7);
                --border-color: rgba(255, 255, 255, 0.08);
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --accent-primary: #818cf8;
                --accent-secondary: #c084fc;
                --user-msg-bg: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                --bot-msg-bg: rgba(255, 255, 255, 0.05);
            }
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Outfit', 'Noto Sans KR', sans-serif;
                background: var(--bg-gradient);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }
            
            .chat-container {
                width: 90%;
                max-width: 800px;
                height: 80vh;
                background: var(--panel-bg);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--border-color);
                border-radius: 24px;
                display: flex;
                flex-direction: column;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                overflow: hidden;
                position: relative;
            }
            
            .chat-header {
                padding: 24px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: rgba(15, 23, 42, 0.4);
            }
            
            .chat-header h1 {
                font-size: 1.25rem;
                font-weight: 600;
                background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .status-badge {
                font-size: 0.75rem;
                padding: 6px 12px;
                border-radius: 20px;
                background: rgba(16, 185, 129, 0.1);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.2);
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #10b981;
                box-shadow: 0 0 8px #10b981;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }
            
            .chat-messages {
                flex: 1;
                padding: 24px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
                scroll-behavior: smooth;
            }
            
            /* Scrollbar styling */
            .chat-messages::-webkit-scrollbar {
                width: 6px;
            }
            .chat-messages::-webkit-scrollbar-track {
                background: transparent;
            }
            .chat-messages::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }
            
            .message {
                max-width: 75%;
                display: flex;
                flex-direction: column;
                gap: 6px;
                animation: fadeIn 0.3s ease-out forwards;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message.user {
                align-self: flex-end;
            }
            
            .message.bot {
                align-self: flex-start;
            }
            
            .message-sender {
                font-size: 0.75rem;
                color: var(--text-secondary);
                padding: 0 4px;
            }
            
            .message-bubble {
                padding: 14px 20px;
                border-radius: 20px;
                font-size: 0.95rem;
                line-height: 1.5;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .message.user .message-bubble {
                background: var(--user-msg-bg);
                color: #ffffff;
                border-bottom-right-radius: 4px;
            }
            
            .message.bot .message-bubble {
                background: var(--bot-msg-bg);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                border-bottom-left-radius: 4px;
            }
            
            .chat-input-area {
                padding: 24px;
                border-top: 1px solid var(--border-color);
                display: flex;
                gap: 12px;
                background: rgba(15, 23, 42, 0.4);
                align-items: center;
            }
            
            .chat-input {
                flex: 1;
                height: 50px;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 0 20px;
                font-family: inherit;
                color: var(--text-primary);
                font-size: 0.95rem;
                outline: none;
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            
            .chat-input:focus {
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
            }
            
            .send-btn {
                height: 50px;
                width: 50px;
                border-radius: 14px;
                background: var(--user-msg-bg);
                border: none;
                color: white;
                cursor: pointer;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: transform 0.1s, opacity 0.2s;
            }
            
            .send-btn:hover {
                opacity: 0.9;
                transform: translateY(-1px);
            }
            
            .send-btn:active {
                transform: translateY(1px);
            }
            
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 4px 8px;
            }
            
            .typing-dot {
                width: 6px;
                height: 6px;
                background: var(--text-secondary);
                border-radius: 50%;
                opacity: 0.4;
                animation: typing 1.4s infinite ease-in-out;
            }
            
            .typing-dot:nth-child(1) { animation-delay: 0s; }
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 100% { transform: translateY(0); opacity: 0.4; }
                50% { transform: translateY(-4px); opacity: 1; }
            }
            
            .parameter-control {
                font-size: 0.75rem;
                color: var(--text-secondary);
                display: flex;
                gap: 12px;
                padding: 4px 24px;
                background: rgba(15, 23, 42, 0.4);
                border-top: 1px solid rgba(255, 255, 255, 0.03);
            }
            
            .param-item {
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .param-item input {
                background: transparent;
                border: none;
                color: var(--accent-primary);
                width: 35px;
                font-weight: 600;
                outline: none;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                    Korean Dense Transformer Chatbot
                </h1>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    Model Online
                </div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    <span class="message-sender">Assistant</span>
                    <div class="message-bubble">
                        안녕하세요! 대화를 시작해보세요. (예: "안녕" 등을 입력해 보세요.)
                    </div>
                </div>
            </div>
            
            <div class="parameter-control">
                <div class="param-item">
                    Temp: <input type="number" step="0.1" min="0.1" max="1.5" id="paramTemp" value="1.0">
                </div>
                <div class="param-item">
                    Top-P: <input type="number" step="0.05" min="0.1" max="1.0" id="paramTopP" value="0.95">
                </div>
                <div class="param-item">
                    Rep Pen: <input type="number" step="0.1" min="1.0" max="2.0" id="paramRep" value="1.5">
                </div>
                <div class="param-item">
                    Max Tokens: <input type="number" step="10" min="10" max="256" id="paramMaxTokens" value="100">
                </div>
            </div>
            
            <div class="chat-input-area">
                <input type="text" class="chat-input" id="chatInput" placeholder="메시지를 입력하세요..." autofocus>
                <button class="send-btn" id="sendBtn">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
        </div>
        
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
                const indicator = document.getElementById('typingIndicator');
                if (indicator) {
                    indicator.remove();
                }
            }
            
            async function handleSend() {
                const text = chatInput.value.trim();
                if (!text) return;
                
                chatInput.value = '';
                appendMessage('User', text, true);
                showTypingIndicator();
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
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
                        appendMessage('Assistant', data.reply || '(답변을 이해하지 못했습니다.)');
                    } else {
                        appendMessage('Assistant', `Error: ${data.detail || 'Failed to generate response'}`);
                    }
                } catch (error) {
                    removeTypingIndicator();
                    appendMessage('Assistant', `Error: ${error.message}`);
                }
            }
            
            sendBtn.addEventListener('click', handleSend);
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    handleSend();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

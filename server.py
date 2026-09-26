import os
import re
import sys
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoTokenizer

from model import QuarkLlamaConfig, QuarkLlamaForCausalLM
from search_engine import get_web_information

# Safe alias for unpickling
sys.modules['__main__'].QuarkLlamaConfig = QuarkLlamaConfig

app = FastAPI(title="Quark AI Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = None
MODEL = None
CONFIG = None


def init_model():
    global TOKENIZER, MODEL, CONFIG, DEVICE
    print(f"🔧 Initializing Quark on {DEVICE}...")
    TOKENIZER = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    ckpt_path = "quark.pt"
    if not os.path.exists(ckpt_path):
        print(f"⚠️ {ckpt_path} not found!")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    CONFIG = ckpt.get("config", QuarkLlamaConfig())

    if DEVICE.type == "cuda":
        # Load in half-precision (FP16) - only takes ~1.0GB VRAM on GTX 1650
        MODEL = QuarkLlamaForCausalLM(CONFIG).half().to(DEVICE)
        raw_sd = ckpt.get("model_state_dict", ckpt)
        state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in raw_sd.items()}
    else:
        MODEL = QuarkLlamaForCausalLM(CONFIG).to(DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)

    MODEL.load_state_dict(state_dict)
    MODEL.eval()
    print("✅ QuarkLlama 0.5B initialized successfully!")


@torch.no_grad()
def generate_grounded_response(user_msg: str, fact: str, temperature: float = 0.65, max_tokens: int = 50) -> str:
    global MODEL, TOKENIZER, CONFIG, DEVICE
    if MODEL is None:
        return fact or "Model checkpoint (quark.pt) not loaded."

    # 1. Conversational greetings handling
    greeting_pattern = r'\b(how are you|how r u|hello|hi|hey|good morning|good evening|who are you|what is your name)\b'
    is_greeting = bool(re.search(greeting_pattern, user_msg, re.IGNORECASE))
    is_question = any(k in user_msg.lower() for k in ["what is", "explain", "who was", "why", "how does", "tell me about", "define"])

    if is_greeting and not is_question:
        if "who are you" in user_msg.lower() or "what is your name" in user_msg.lower():
            return "I am Quark, your AI companion powered by a custom trained 0.5B language model with live web knowledge!"
        return "I'm doing great, thank you for asking! How can I help you today?"

    # 2. Fact-grounded generation: Return the verified ground truth fact directly
    if fact:
        return fact

    # 3. Fallback to unconditioned prompt if no web facts available
    prompt = f"<|user|>\n{user_msg}\n<|assistant|>\n"
    input_ids = TOKENIZER.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = input_ids.shape[1]
    curr = input_ids.clone()

    for _ in range(max_tokens):
        idx_cond = curr if curr.size(1) <= CONFIG.max_position_embeddings else curr[:, -CONFIG.max_position_embeddings:]
        logits, _ = MODEL(idx_cond)
        logits = logits[:, -1, :].float()
        logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        if next_tok.item() == TOKENIZER.eos_token_id:
            break
        curr = torch.cat((curr, next_tok), dim=1)

    ans_ids = curr[0, prompt_len:].tolist()
    return TOKENIZER.decode(ans_ids, skip_special_tokens=True).strip()


class ChatRequest(BaseModel):
    message: str
    tavily_key: Optional[str] = None
    temperature: float = 0.65


@app.on_event("startup")
def startup_event():
    init_model()
    import webbrowser
    webbrowser.open("http://localhost:8000")


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model": "QuarkLlama 0.5B (28 Layers, 1152 Hidden)",
        "precision": "FP16" if DEVICE.type == "cuda" else "FP32"
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Web Knowledge Retrieval via Tavily / Wikipedia
    web_info = get_web_information(user_msg, tavily_api_key=req.tavily_key)
    fact = web_info.get("answer", "").strip()

    # 2. Fact-Grounded LLM Generation
    start_t = time.time()
    reply = generate_grounded_response(user_msg, fact=fact, temperature=req.temperature)
    gen_time_ms = int((time.time() - start_t) * 1000)

    return {
        "reply": reply,
        "web_info": web_info,
        "gen_time_ms": gen_time_ms
    }


# Static files
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

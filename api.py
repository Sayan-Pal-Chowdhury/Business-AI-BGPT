import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.tinygpt import CharTokenizer, GPTConfig, TinyGPT


CHECKPOINT = Path("checkpoints/bgpt.pt")
PUBLIC_DIR = Path("public")
KNOWLEDGE_BASE = Path("data/knowledge_base.json")
app = FastAPI(title="Business AI BGPT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

model = None
tokenizer = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 20


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 120
    temperature: float = 0.75
    top_k: int = 20
    mode: str = "hybrid"
    include_core_sample: bool = True


BUSINESS_KEYWORDS = {
    "business", "shop", "shopkeeper", "sales", "sale", "inventory", "stock",
    "credit", "customer", "customers", "profit", "loss", "cash", "payment",
    "delivery", "order", "orders", "expense", "expenses", "revenue", "billing",
    "supplier", "product", "products", "operations", "wholesale", "retail",
    "zuno", "stapo", "analytics", "dashboard", "invoice", "margin", "price",
}


def is_business_question(message):
    lowered = message.lower()
    return any(keyword in lowered for keyword in BUSINESS_KEYWORDS)


def tokenize_words(text):
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {word for word in cleaned.split() if len(word) > 2}


def load_knowledge_base():
    if not KNOWLEDGE_BASE.exists():
        return []
    return json.loads(KNOWLEDGE_BASE.read_text(encoding="utf-8"))


def retrieve_context(message, limit=3):
    query_terms = tokenize_words(message)
    scored = []
    for item in load_knowledge_base():
        haystack = " ".join([
            item.get("title", ""),
            item.get("content", ""),
            " ".join(item.get("tags", [])),
        ])
        overlap = len(query_terms & tokenize_words(haystack))
        if overlap:
            scored.append((overlap, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored[:limit]]


def build_bgpt_prompt(message, contexts):
    context_text = "\n".join(f"- {item['content']}" for item in contexts)
    return (
        "Sayan: You are BGPT, a Business AI model trained only for business/shopkeeper workflows.\n"
        f"Business context:\n{context_text}\n"
        f"User: {message.strip()}\n"
        "BGPT:"
    )


def generate_with_bgpt(prompt, max_new_tokens, temperature, top_k):
    loaded_model, loaded_tokenizer = load_checkpoint()
    ids = loaded_tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = loaded_model.generate(
            idx,
            max_new_tokens=max(1, min(300, max_new_tokens)),
            temperature=temperature,
            top_k=max(1, min(100, top_k)),
        )[0].tolist()
    return loaded_tokenizer.decode(out[len(ids):]).strip()


def compose_business_reply(message, contexts):
    if not contexts:
        return (
            "BGPT can help with business workflows such as sales, inventory, credit, "
            "profit, cash flow, customers, orders, billing, delivery, and operations."
        )

    bullets = [item["content"] for item in contexts[:3]]
    lowered = message.lower()
    if "credit" in lowered or "late" in lowered or "payment" in lowered:
        lead = "For credit handling, focus on risk, reminders, and limits."
    elif "stock" in lowered or "inventory" in lowered or "product" in lowered:
        lead = "For inventory, focus on fast-moving products and reorder timing."
    elif "profit" in lowered or "loss" in lowered or "margin" in lowered:
        lead = "For profit, compare buying price, selling price, expenses, discounts, and unpaid credit."
    elif "cash" in lowered or "expense" in lowered:
        lead = "For cash flow, collect pending credit and control unnecessary expenses."
    else:
        lead = "Here is a business-focused answer from BGPT's retrieved knowledge."

    return lead + "\n\n" + "\n".join(f"- {bullet}" for bullet in bullets)


def call_gemini_fallback(message, contexts):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    context_text = "\n".join(f"- {item['content']}" for item in contexts)
    prompt = (
        "You are BGPT, Business AI. Answer only business questions about sales, "
        "inventory, credit, profit, cash flow, customers, orders, billing, delivery, "
        "or operations. Be short and practical.\n\n"
        f"Retrieved business context:\n{context_text}\n\n"
        f"Question: {message}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": 220},
    }).encode("utf-8")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=18) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
        return None

    candidates = payload.get("candidates", [])
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip() or None


def load_checkpoint():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return model, tokenizer
    if not CHECKPOINT.exists():
        raise RuntimeError("No checkpoint found. Run train.py first.")
    payload = torch.load(CHECKPOINT, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    config = GPTConfig(**payload["config"])
    model = TinyGPT(config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer


@app.get("/health")
def health():
    return {
        "ok": True,
        "checkpoint": CHECKPOINT.exists(),
        "model": "business-ai-bgpt",
        "rag": KNOWLEDGE_BASE.exists(),
        "fallback_llm": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.get("/")
def index():
    index_path = PUBLIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"ok": True, "message": "ZunoTinyGPT API is running."}


@app.post("/generate")
def generate(req: GenerateRequest):
    loaded_model, loaded_tokenizer = load_checkpoint()
    ids = loaded_tokenizer.encode(req.prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = loaded_model.generate(
            idx,
            max_new_tokens=max(1, min(300, req.max_new_tokens)),
            temperature=req.temperature,
            top_k=max(1, min(100, req.top_k)),
        )[0].tolist()
    return {
        "prompt": req.prompt,
        "text": loaded_tokenizer.decode(out),
        "model": "business-ai-bgpt",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if not is_business_question(req.message):
        return {
            "reply": (
                "BGPT is business-only. Ask about sales, inventory, credit, "
                "profit, customers, orders, billing, delivery, or operations."
            ),
            "full_text": "",
            "model": "business-ai-bgpt",
            "note": "Business-only guardrail response.",
        }

    contexts = retrieve_context(req.message)
    prompt = build_bgpt_prompt(req.message, contexts)
    core_reply = (
        generate_with_bgpt(prompt, req.max_new_tokens, req.temperature, req.top_k)
        if req.include_core_sample else ""
    )
    fallback_reply = call_gemini_fallback(req.message, contexts) if req.mode == "hybrid" else None
    reply = fallback_reply or compose_business_reply(req.message, contexts)

    return {
        "reply": core_reply if req.mode == "bgpt" else reply,
        "bgpt_core_sample": core_reply,
        "retrieved_context": contexts,
        "model": "business-ai-bgpt",
        "mode": "gemini-rag-fallback" if fallback_reply else req.mode,
        "note": "BGPT combines a tiny transformer core with business RAG and optional LLM fallback.",
    }

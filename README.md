---
title: Business AI BGPT
emoji: 🧠
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Business AI BGPT

Business AI BGPT is a small GPT-style Transformer project for learning LLM internals and later connecting an experimental business-only model back into Zuno.

This is not meant to compete with Gemini or ChatGPT. It is a learning model that helps demonstrate:

- tokenization
- embeddings
- positional embeddings
- causal self-attention
- transformer blocks
- next-token prediction
- training loop
- inference API
- RAG-style business context retrieval
- optional production LLM fallback

## Project Goal

Build a tiny business-domain model trained on shopkeeper/business text, wrap it with business RAG and optional LLM fallback, deploy it as an API/frontend, and use it inside Zuno as an experimental business AI provider.

Good interview wording:

> I built BGPT, a small business-only GPT-style Transformer from scratch in PyTorch to understand LLM internals, trained it on shop/business text, exposed it through an API and ChatGPT-like frontend, and planned integration into my Zuno business app as an experimental model.

## Architecture

- **BGPT Core:** character-level GPT-style Transformer built from scratch in PyTorch.
- **Business Guardrail:** accepts only business/shopkeeper/operations questions.
- **RAG Layer:** retrieves relevant business context from `data/knowledge_base.json`.
- **Optional LLM Fallback:** if `GEMINI_API_KEY` is set, hybrid mode can use Gemini for a stronger final response while still showing BGPT core output.
- **Frontend:** ChatGPT-like UI for business-only questions.

## Setup

Recommended Python: 3.11 or 3.12.

Your current Python is 3.14, and PyTorch may not install cleanly on it yet. If installation fails, install Python 3.11/3.12 and create a virtual environment with that version.

```powershell
cd C:\Users\sayan\OneDrive\Desktop\ZunoTinyGPT
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```powershell
python train.py --data data/business_seed.txt --steps 500
```

## Generate Text

```powershell
python generate.py --prompt "customer paid late"
```

## Run API

```powershell
uvicorn api:app --reload --port 8008
```

Test:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8008/chat" -Method POST -ContentType "application/json" -Body '{"message":"How should a shop handle pending customer credit?","max_new_tokens":80}' | Select-Object -ExpandProperty Content
```

Optional stronger fallback:

```powershell
$env:GEMINI_API_KEY="your_key_here"
uvicorn api:app --reload --port 8008
```

## Deploy

Recommended first host: Hugging Face Spaces.

Why Hugging Face Spaces:

- AI-native portfolio platform
- free CPU Basic Spaces are suitable for this small checkpoint
- supports Docker, so the FastAPI + custom frontend can run cleanly
- can store `GEMINI_API_KEY` as a Space secret later

The hosted Docker build trains the small BGPT checkpoint during image creation. Local checkpoints stay ignored by Git.

## Roadmap

1. Train a tiny character-level GPT locally.
2. Improve dataset with shopkeeper/business language.
3. Deploy API and ChatGPT-like frontend.
4. Add a hosted vector DB such as Qdrant when the knowledge base grows.
5. Add LangChain orchestration if the project needs multi-step tools/chains.
6. Add Zuno backend route that can call BGPT as an experimental provider.
7. Add a Zuno UI toggle: Gemini vs BGPT experiment.

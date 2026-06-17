# Project Story

## One-line Summary

Business AI BGPT is a small business-only GPT-style Transformer built from scratch in PyTorch to understand LLM internals and later connect an experimental domain model into Zuno.

## Why I Built It

After integrating LLMs, RAG, embeddings, Qdrant vector database, and LangChain into Zuno, the next step was to learn how language models work internally.

This project focuses on the fundamentals behind LLMs:

- tokenization
- embeddings
- positional encodings
- causal self-attention
- transformer blocks
- next-token prediction
- model training and inference

## How It Connects To Zuno

Zuno uses production AI APIs for useful business workflows.
BGPT is the learning and experimentation side: a small model trained on shopkeeper/business text, wrapped with business RAG retrieval and optional production LLM fallback, exposed as an API and ChatGPT-like frontend, and later connected to Zuno as an experimental model provider.

## Interview Explanation

> I first integrated production AI into my business app using Gemini, RAG, embeddings, Qdrant, and LangChain. Then I built BGPT, a small business-only GPT-style Transformer from scratch in PyTorch to understand the internals behind LLMs, including attention, embeddings, training loops, and inference. Around the core model I added business guardrails, RAG-style retrieval, and optional LLM fallback, then exposed it through an API and frontend.

## What This Project Is Not

This is not a replacement for Gemini or ChatGPT.
It is a small educational business-domain model for learning and demonstrating AI fundamentals.

## Next Milestones

1. Train the tiny model locally.
2. Expand the business-domain dataset.
3. Add evaluation prompts.
4. Expose inference through FastAPI and a ChatGPT-like frontend.
5. Upgrade the RAG layer to a hosted vector DB when the knowledge base grows.
6. Add LangChain orchestration if multi-step tools are needed.
7. Deploy the API on an always-on platform such as Railway.
8. Add Zuno backend integration as an experimental business model.

# RTFM For Me — AI Documentation Assistant

An AI-powered documentation assistant built with RAG (Retrieval-Augmented Generation), implemented as part of [Coding Challenge #110: RTFM For Me](https://codingchallenges.substack.com/p/coding-challenge-110-rtfm-for-me).

## Overview

The assistant answers questions about a library by searching relevant documentation and generating grounded responses using vector search and LLMs.

## Demo Documentation

Uses documentation from [FusionCache](https://github.com/ZiggyCreatures/FusionCache) for demo purposes only. FusionCache is an advanced caching library for .NET — all credit and ownership of the docs belongs to the FusionCache project.

## Stack

- **LLM & Embeddings**: OpenAI
- **Framework**: LangChain
- **Vector Store**: Redis Stack (via `redisvl`)

## Setup

```bash
# Start Redis
docker compose up -d

# Install dependencies
pip install -r requirements.txt

# Ingest documentation
python ingest.py
```

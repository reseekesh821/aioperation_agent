# AI Operations Agent

A production-oriented learning project: a grounded AI agent that answers inventory, order, refund, and policy questions using FastAPI, LangGraph, Ollama, ChromaDB, and PostgreSQL.

The LLM proposes answers; Python enforces business rules, tool permissions, and factual grounding before anything is returned to the user.

## Features

- Inventory lookup with product-name normalization (singular/plural aliases)
- Order status lookup from PostgreSQL
- Refund eligibility checks with delivery and 30-day rules
- Policy Q&A via RAG over `data/policies/refund_policy.txt`
- Grounded natural-language responses with deterministic validation and safe fallbacks
- HTTP 503 when Ollama is unreachable
- Simple vanilla JavaScript chat UI

## Architecture

```
Browser (frontend :5500)
    → FastAPI /chat (backend :8000)
        → LangGraph agent
            → Tools: inventory | orders | refunds | policy retrieval
            → PostgreSQL (orders, inventory)
            → ChromaDB + nomic-embed-text (policy RAG)
            → llama3.1:8b (structured planning + natural answers)
            → Python grounding validation → answer or verified fallback
```

## Technology stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Agent | LangGraph + LangChain tools |
| LLM / embeddings | Ollama (`llama3.1:8b`, `nomic-embed-text`) |
| Database | PostgreSQL 18 |
| Vector store | ChromaDB |
| Frontend | HTML / CSS / vanilla JS |
| Packaging | Docker + Docker Compose |

## Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.com) running on the host (port 11434)
- Python 3.14+ (for local tests or serving the frontend)
- Pulled models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Quick start (Docker Compose)

1. **Configure environment**

```bash
cp .env
# Edit .env and set POSTGRES_PASSWORD (and matching DATABASE_URL password)
```

2. **Build the backend image**

```bash
docker build -t ai-ops-backend:latest .
```

3. **Start Postgres + backend**

```bash
docker compose up -d
```

- API: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- Postgres: `127.0.0.1:5432` (user `aiops`, database `aiops`)

On first start only, Compose mounts `infra/init.sql` into Postgres’s init directory and seeds sample orders and inventory. Existing volumes are left untouched; init scripts do **not** re-run against an already-initialized database.

4. **Start the frontend** (port 5500 — required by CORS)

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500` and ask questions such as:

- How many monitors are available?
- What is the status of order 1003?
- Is order 1001 eligible for a refund?
- What is the refund deadline?

## API overview

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/orders/{order_id}` | Order lookup |
| `GET` | `/inventory/{product_name}` | Inventory lookup |
| `POST` | `/chat` | Agent chat (`{"message": "..."}` → `{"answer": "..."}`) |

## Local tests

With Compose Postgres running and `.env` loaded:

```bash
set -a && source .env && set +a
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -q
```

Ollama must be available for live RAG / generation tests.

## Project layout

```
backend/app/          FastAPI app, LangGraph agent, services
data/policies/        Authoritative refund policy text
data/chroma/          Generated vector index (gitignored)
frontend/             Chat UI
infra/init.sql        Fresh-database schema + seed data
compose.yaml          Postgres + backend services
Dockerfile            Backend image
```

## Notes

- The backend reaches host Ollama via `host.docker.internal`.
- ChromaDB data is stored under `data/chroma/` and recreated from policy text when empty.
- Sample JSON files under `data/` mirror the seed rows for reference; runtime reads come from PostgreSQL.

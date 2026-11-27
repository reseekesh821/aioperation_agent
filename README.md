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
Customer portal (:5000)  ─┐
                          ├→ FastAPI (backend :8000)
Owner workspace (:5001)  ─┘
        → LangGraph agent + Gmail/support case services
            → PostgreSQL (orders, inventory, olist, support)
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

4. **Start the frontends** (exact ports required by CORS)

From the project root:

```bash
python -m http.server 5000 --bind 127.0.0.1 --directory frontend/customer
```

```bash
python -m http.server 5001 --bind 127.0.0.1 --directory frontend/owner
```

- Customer: `http://127.0.0.1:5000/`
- Owner: `http://127.0.0.1:5001/` (Inbox / Cases views)

Example customer questions:

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
frontend/customer/    Customer chatbot portal (:5000)
frontend/owner/       Owner support workspace (:5001)
infra/init.sql        Fresh-database schema + seed data
infra/support_schema.sql  Support case schema (apply separately if needed)
compose.yaml          Postgres + backend services
Dockerfile            Backend image
```

## Notes

- The backend reaches host Ollama via `host.docker.internal`.
- ChromaDB data is stored under `data/chroma/` and recreated from policy text when empty.
- Sample JSON files under `data/` mirror the seed rows for reference; runtime reads come from PostgreSQL.
- CORS allows only `http://127.0.0.1:5000` and `http://127.0.0.1:5001` (no wildcard).
- Serve each app directory as the HTTP root so `/` opens the app directly.

## Security limitations (localhost development)

- Separate frontend pages and CORS are **not** authentication or authorization.
- Owner API routes are reachable by any localhost client that can call port 8000.
- Customer identity / order ownership is not verified for public deployment.
- Gmail OAuth tokens stay on the backend (`token.json`); do not place credentials in static frontend files.
- Before any public deployment, add owner authentication and customer identity verification.

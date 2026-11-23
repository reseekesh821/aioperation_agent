import chromadb
from pathlib import Path
from ollama import embed

from backend.app.services.policies import get_policy_chunks, get_policy_embeddings

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "chroma"

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name="refund_policies")


def store_policies():
    chunks = get_policy_chunks()
    embeddings = get_policy_embeddings()

    collection.upsert(
        ids=[f"policy_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )

    return collection.count()


def ensure_policies_indexed() -> int:
    """Seed Chroma when empty so retrieval does not silently fail."""
    if collection.count() == 0:
        return store_policies()
    return collection.count()


def search_policies(question: str) -> list[str]:
    ensure_policies_indexed()

    response = embed(
        model="nomic-embed-text",
        input=question,
    )

    results = collection.query(
        query_embeddings=response.embeddings,
        n_results=2,
    )

    documents = results.get("documents") or [[]]
    return documents[0] or []

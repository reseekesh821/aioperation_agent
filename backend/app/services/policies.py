from pathlib import Path
from ollama import embed

POLICY_PATH = Path(__file__).resolve().parents[3] / "data" / "policies" / "refund_policy.txt"

def load_refund_policy() -> str:
    return POLICY_PATH.read_text(encoding="utf-8")

def get_policy_chunks() -> list[str]:
    policy = load_refund_policy()

    return [
        line.strip()
        for line in policy.splitlines()
        if line.strip() and line.strip() != "REFUND POLICY"
    ]
def get_policy_embeddings():
    chunks = get_policy_chunks()

    response = embed(
        model="nomic-embed-text",
        input=chunks
    )

    return response.embeddings
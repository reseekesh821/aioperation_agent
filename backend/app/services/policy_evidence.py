from backend.app.services.policies import get_policy_chunks
from backend.app.services.vector_store import search_policies


def load_policy_records() -> dict[str, str]:
    records = {}

    for line in get_policy_chunks():
        number, separator, text = line.partition(". ")

        if not separator or not number.isdigit() or not text:
            raise ValueError(f"Invalid policy rule: {line}")

        rule_id = f"refund_{number}"

        if rule_id in records:
            raise ValueError(f"Duplicate policy ID: {rule_id}")

        records[rule_id] = text

    return records


def retrieve_policy_evidence(
    question: str,
    chunks: list[str] | None = None,
) -> list[dict]:
    records = load_policy_records()

    source_map = {
        f"{rule_id.split('_')[1]}. {text}": {
            "id": rule_id,
            "text": text,
        }
        for rule_id, text in records.items()
    }

    retrieved = search_policies(question) if chunks is None else chunks

    return [
        source_map[chunk]
        for chunk in retrieved
        if chunk in source_map
    ]


# Keyword signals boost/rank retrieved rules; they are not the only gate.
RULE_SIGNALS = {
    "refund_1": (
        "deadline",
        "how long",
        "30 days",
        "time limit",
        "within",
        "after delivery",
    ),
    "refund_2": (
        "eligibility",
        "eligible",
        "delivered",
        "qualify",
        "standard refund",
    ),
    "refund_3": (
        "shipping",
        "shipped",
        "processing",
        "before delivery",
        "while shipping",
        "in transit",
    ),
    "refund_4": (
        "damaged",
        "defective",
        "broken",
        "support review",
    ),
    "refund_5": (
        "approval",
        "approved",
        "guaranteed",
        "guarantee",
        "does not guarantee",
    ),
}

# Topics the refund policy never covers. Retrieval may still return nearby
# chunks; we drop evidence so the planner cannot invent coverage.
UNSUPPORTED_POLICY_TOPICS = (
    "warranty",
    "lifetime warranty",
    "store credit",
    "gift card",
    "exchange",
    "replacement",
    "price match",
)


def is_unsupported_policy_topic(question: str) -> bool:
    q = question.casefold()
    return any(topic in q for topic in UNSUPPORTED_POLICY_TOPICS)


def select_policy_evidence(
    question: str,
    chunks: list[str] | None = None,
) -> list[dict]:
    retrieved = retrieve_policy_evidence(question, chunks)
    if not retrieved:
        return []

    query = question.casefold()

    if is_unsupported_policy_topic(query):
        return []

    scored: list[tuple[int, dict]] = []
    for evidence in retrieved:
        signals = RULE_SIGNALS.get(evidence["id"], ())
        score = sum(1 for signal in signals if signal in query)
        scored.append((score, evidence))

    matched = [item for score, item in scored if score > 0]
    if matched:
        # Prefer higher signal overlap while preserving retrieval order ties.
        matched.sort(
            key=lambda item: sum(
                1 for signal in RULE_SIGNALS.get(item["id"], ())
                if signal in query
            ),
            reverse=True,
        )
        return matched

    # No keyword hit: keep retrieved evidence and let the claim planner decide.
    # This avoids false refusals when phrasing differs from RULE_SIGNALS.
    return [item for _, item in scored]


APPROVED_CLAIMS = {
    "refund_1": (
        "Standard refund requests must be submitted within 30 days after delivery."
    ),
    "refund_2": (
        "Only delivered orders qualify for standard refund review."
    ),
    "refund_3": (
        "Orders still shipping or processing cannot submit a standard refund request."
    ),
    "refund_4": (
        "Damaged or defective products require a support review."
    ),
    "refund_5": (
        "Refund eligibility does not guarantee refund approval."
    ),
}

NO_POLICY_INFO = "The available policies do not specify this information."

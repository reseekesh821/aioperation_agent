"""Shared generate → validate → (retry) → fallback response layer."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.services.inventory import PRODUCT_ALIASES, list_products
from backend.app.services.policy_evidence import NO_POLICY_INFO
from backend.app.agents.ai_errors import is_ai_service_error

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """
You are a helpful operations assistant.

Answer the user's question naturally in one or two short sentences.
Use only the verified facts provided below.
Do not invent quantities, statuses, products, policies, exceptions, or promises.
Do not add product adjectives, attributes, grades, or specifications that are not
explicitly present in the verified facts (for example, do not turn "Mouse" into
"standard mouse" or "wireless mouse").
Preserve important numbers, product names, order IDs, statuses, and qualifications
such as "standard" only when they appear in the verified facts.
Do not mention tools, JSON, claim IDs, or say that you are rewriting an answer.
If the facts say the policies do not specify the information, say that clearly
without inventing coverage.
"""

RETRY_SYSTEM_PROMPT = """
Your previous draft was rejected because it changed or dropped verified facts.

Rewrite the answer in one or two short sentences using only the verified facts.
Keep every important number, product name, order ID, status, and qualification
from the verified facts. Do not invent anything new.
Do not add unsupported product adjectives or specifications.
Do not mention rewriting, drafts, or instructions.
"""

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "to", "of", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being", "by", "with", "as",
    "at", "from", "that", "this", "these", "those", "it", "its", "you", "your",
    "we", "our", "they", "their", "can", "could", "should", "would", "may",
    "might", "must", "do", "does", "did", "not", "no", "yes", "i", "me",
    "my", "will", "just", "than", "then", "so", "such", "only", "also",
    "about", "into", "over", "after", "before", "while", "when", "what",
    "which", "who", "whom", "how", "any", "all", "some", "please",
}

STYLE_ALLOWLIST = {
    "currently", "there", "here", "have", "has", "available", "stock",
    "units", "unit", "left", "remaining", "shows", "show", "status",
    "found", "catalog", "sorry", "looks", "like", "still", "under",
    "review", "request", "requests", "policy", "policies", "company",
    "offer", "offered", "information", "specify", "specified", "covered",
    "see", "appears", "according", "verified", "result", "results",
    "question", "answer", "yes", "no", "cannot", "can", "don", "doesn",
    "within", "days", "day", "after", "before", "delivery", "delivered",
    "shipping", "processing", "shipped", "order", "orders", "product",
    "products", "refund", "refunds", "eligible", "eligibility", "standard",
    "submit", "submitted", "unavailable", "approval", "approved",
    "guarantee", "guaranteed", "damaged", "defective", "support",
    "require", "requires", "qualify", "qualifies", "deadline", "sure",
    "right", "now", "however", "therefore", "mention", "mentions",
    "mentioned", "because", "been", "being", "rewritten",
}

HIGH_RISK_PHRASES = (
    "store credit",
    "gift card",
    "lifetime warranty",
    "warranty",
    "exchange",
    "replacement",
    "price match",
    "full refund",
    "partial refund",
    "money back",
    "instant refund",
    "free shipping",
)

UNCERTAINTY_MARKERS = (
    "do not specify",
    "does not specify",
    "don't specify",
    "not specify",
    "not specified",
    "no information",
    "not covered",
    "do not cover",
    "doesn't cover",
    "do not mention",
    "don't mention",
    "does not mention",
    "doesn't mention",
    "not mention",
    "not available in",
    "no policy",
)

NEGATION_CUES = (
    "not",
    "no ",
    "don't",
    "do not",
    "doesn't",
    "does not",
    "never",
    "without",
    "unavailable",
)

STATUS_PATTERNS = (
    ("out of stock", ("out of stock", "no .* in stock", "don't have any", "do not have any")),
    ("not in the catalog", ("not in the catalog", "not in our catalog", "unknown product")),
    ("not found", ("not found", "could not find", "no order")),
    ("delivered", ("delivered",)),
    ("shipped", ("shipped",)),
    ("processing", ("processing",)),
    (
        "eligible for refund review",
        ("eligible for refund review", "eligible for a refund review"),
    ),
    (
        "refund review unavailable",
        ("refund review unavailable", "not eligible", "unavailable"),
    ),
)


@dataclass
class GeneratedResponse:
    answer: str
    used_fallback: bool
    attempts: int
    reason: str | None = None
    candidates: list[str] = field(default_factory=list)


_LAST_GENERATION: GeneratedResponse | None = None


def get_last_generation() -> GeneratedResponse | None:
    return _LAST_GENERATION


def _set_last_generation(result: GeneratedResponse) -> GeneratedResponse:
    global _LAST_GENERATION
    _LAST_GENERATION = result
    if result.used_fallback:
        logger.info(
            "response_generation fallback after %s attempt(s): %s",
            result.attempts,
            result.reason or "validation_failed",
        )
    else:
        logger.info(
            "response_generation accepted LLM answer after %s attempt(s)",
            result.attempts,
        )
    return result


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in STOPWORDS]


def _message_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return (content or "").strip()


def _clean_candidate(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return cleaned

    preface = lines[0].casefold()
    if (
        "rewritten answer" in preface
        or preface.startswith("here is a rewritten")
        or preface.startswith("here is the rewritten")
    ):
        lines = lines[1:] or lines

    return " ".join(lines).strip()


def _product_aliases(canonical: str) -> set[str]:
    names = {canonical.casefold()}
    for alias, target in PRODUCT_ALIASES.items():
        if target.casefold() == canonical.casefold():
            names.add(alias.casefold())
    return names


def _mentions_product(answer_lower: str, canonical: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(alias)}\b", answer_lower)
        for alias in _product_aliases(canonical)
    )


# Grammatical words that may precede a product without adding an attribute.
SAFE_PRE_PRODUCT = {
    "a", "an", "the", "our", "your", "their", "this", "that", "any", "some",
    "one", "each", "per", "for", "of", "on", "with", "and", "or", "to", "from",
    "about", "available", "remaining", "said", "named",
}


def _product_surface_forms(canonical: str) -> set[str]:
    """Single- and multi-word surface forms for a product mention."""
    forms = set(_product_aliases(canonical))
    forms.add(canonical.casefold())
    return forms


def _has_unsupported_product_attributes(
    answer: str,
    verified_facts: str,
    products: set[str],
) -> bool:
    """
    Reject invented product adjectives/specs adjacent to product names.

    A pre-product modifier is allowed only when verified facts already contain
    that same "<modifier> <product>" pairing (or the modifier is grammatical).
    """
    if not products:
        return False

    answer_lower = answer.casefold()
    facts_lower = verified_facts.casefold()

    for product in products:
        for form in sorted(_product_surface_forms(product), key=len, reverse=True):
            pattern = rf"\b([a-z0-9][\w-]*)\s+{re.escape(form)}\b"
            for match in re.finditer(pattern, answer_lower):
                modifier = match.group(1)
                if modifier in SAFE_PRE_PRODUCT or modifier.isdigit():
                    continue
                if modifier in _product_surface_forms(product):
                    continue
                # Allow only if facts already use this modifier with this product.
                if re.search(
                    rf"\b{re.escape(modifier)}\s+{re.escape(form)}\b",
                    facts_lower,
                ):
                    continue
                return True

    return False


def _phrase_negated(answer_lower: str, phrase: str) -> bool:
    idx = answer_lower.find(phrase)
    if idx < 0:
        return False
    window = answer_lower[max(0, idx - 48):idx]
    return any(cue in window for cue in NEGATION_CUES)


def extract_required_anchors(verified_facts: str) -> dict[str, set[str]]:
    """Deterministic anchors grouped by role."""
    facts = verified_facts.casefold()
    anchors = {
        "numbers": set(re.findall(r"\d+", verified_facts)),
        "statuses": set(),
        "products": set(),
        "qualifiers": set(),
    }

    for status, _ in STATUS_PATTERNS:
        if status in facts:
            anchors["statuses"].add(status)

    for product in list_products():
        if product.casefold() in facts:
            anchors["products"].add(product.casefold())

    # Capture order line products that may not be inventory catalog keys.
    for match in re.finditer(
        r"\border\s+\d+\s*:\s*(.+?)\s*-\s*",
        verified_facts,
        flags=re.IGNORECASE,
    ):
        product_name = match.group(1).strip().casefold()
        if product_name:
            anchors["products"].add(product_name)

    if "standard" in facts:
        anchors["qualifiers"].add("standard")

    return anchors


def _status_satisfied(answer_lower: str, status: str) -> bool:
    for name, patterns in STATUS_PATTERNS:
        if name != status:
            continue
        return any(re.search(pattern, answer_lower) for pattern in patterns)
    return status in answer_lower


def is_response_grounded(
    answer: str,
    verified_facts: str,
    required_anchors: dict[str, set[str]] | None = None,
) -> bool:
    """Python-only grounding check. Not an LLM verifier."""
    if not isinstance(answer, str) or not answer.strip():
        return False

    answer = _clean_candidate(answer)
    answer_lower = answer.casefold()
    facts_lower = verified_facts.casefold()
    anchors = required_anchors or extract_required_anchors(verified_facts)

    if verified_facts.strip() == NO_POLICY_INFO:
        if not any(marker in answer_lower for marker in UNCERTAINTY_MARKERS):
            return False
        for phrase in HIGH_RISK_PHRASES:
            if phrase in answer_lower and not _phrase_negated(answer_lower, phrase):
                return False
        # No invented concrete policy quantities in an uncertainty answer.
        if re.search(r"\b\d+\b", answer):
            return False
        return True

    for phrase in HIGH_RISK_PHRASES:
        if phrase in answer_lower and phrase not in facts_lower:
            return False

    answer_numbers = set(re.findall(r"\d+", answer))
    fact_numbers = set(anchors["numbers"])
    if not answer_numbers.issubset(set(re.findall(r"\d+", verified_facts))):
        return False

    for number in fact_numbers:
        if number in answer_numbers:
            continue
        # Zero stock may be stated as "out of stock" without repeating "0".
        if (
            number == "0"
            and "out of stock" in facts_lower
            and _status_satisfied(answer_lower, "out of stock")
        ):
            continue
        return False

    for status in anchors["statuses"]:
        if not _status_satisfied(answer_lower, status):
            return False

    # Inventory answers must keep the product; order/refund may omit it.
    inventory_like = "units available" in facts_lower or "out of stock" in facts_lower
    if inventory_like:
        for product in anchors["products"]:
            if product in {p.casefold() for p in list_products()}:
                if not _mentions_product(answer_lower, product):
                    return False

    if _has_unsupported_product_attributes(
        answer,
        verified_facts,
        anchors["products"],
    ):
        return False

    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False

    fact_tokens = set(_content_tokens(verified_facts))
    answer_token_set = set(answer_tokens)
    alias_tokens = set()
    for product in anchors["products"]:
        alias_tokens |= _product_aliases(product)

    # Qualifications like "standard" should normally appear. If omitted, require
    # strong overlap with the verified fact tokens so meaning is still preserved.
    for qualifier in anchors["qualifiers"]:
        if qualifier in answer_lower:
            continue
        if not fact_tokens:
            return False
        overlap = sum(1 for token in fact_tokens if token in answer_token_set)
        if overlap / len(fact_tokens) < 0.45:
            return False

    allowed = fact_tokens | STYLE_ALLOWLIST | alias_tokens
    supported = sum(1 for token in answer_tokens if token in allowed)
    if supported / len(answer_tokens) < 0.65:
        return False

    return True


def _invoke_candidate(
    question: str,
    verified_facts: str,
    llm,
    *,
    retry: bool,
    previous: str | None = None,
) -> str:
    system = RETRY_SYSTEM_PROMPT if retry else RESPONSE_SYSTEM_PROMPT
    human_parts = [
        f"Question: {question}",
        f"Verified facts: {verified_facts}",
    ]
    if previous:
        human_parts.append(f"Rejected draft: {previous}")

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content="\n".join(human_parts)),
    ])
    return _clean_candidate(_message_text(response.content))


def generate_grounded_response(
    question: str,
    verified_facts: str,
    llm,
    *,
    max_attempts: int = 2,
) -> GeneratedResponse:
    """
    Generate a natural answer from verified facts.

    Flow: generate → validate → one controlled retry → safe fallback.
    Never returns an unverified candidate.
    """
    facts = (verified_facts or "").strip()
    if not facts:
        return _set_last_generation(
            GeneratedResponse(
                answer=NO_POLICY_INFO,
                used_fallback=True,
                attempts=0,
                reason="empty_verified_facts",
            )
        )

    anchors = extract_required_anchors(facts)
    candidates: list[str] = []
    last_reason = "validation_failed"
    previous = None
    attempts = min(max(max_attempts, 1), 2)

    for attempt in range(1, attempts + 1):
        try:
            candidate = _invoke_candidate(
                question,
                facts,
                llm,
                retry=attempt > 1,
                previous=previous,
            )
        except Exception as exc:  # noqa: BLE001 - connectivity re-raised; others fall back
            if is_ai_service_error(exc):
                raise
            last_reason = f"generation_error:{type(exc).__name__}"
            logger.warning("response_generation LLM error: %s", exc)
            break

        candidates.append(candidate)
        previous = candidate

        if is_response_grounded(candidate, facts, anchors):
            return _set_last_generation(
                GeneratedResponse(
                    answer=candidate,
                    used_fallback=False,
                    attempts=attempt,
                    candidates=candidates,
                )
            )

        last_reason = "validation_failed"

    return _set_last_generation(
        GeneratedResponse(
            answer=facts,
            used_fallback=True,
            attempts=len(candidates),
            reason=last_reason,
            candidates=candidates,
        )
    )

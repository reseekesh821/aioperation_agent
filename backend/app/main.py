import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.orders import get_order
from backend.app.services.inventory import get_inventory, normalize_product_name
from backend.app.agents.agent_graph import agent_graph
from backend.app.agents.ai_errors import is_ai_service_error
from backend.app.integrations.gmail_drafts import (
    GmailDraftValidationError,
    save_support_reply_draft,
)
from backend.app.integrations.gmail_reader import (
    GmailApiError,
    GmailAuthError,
    GmailLabelNotFoundError,
    GmailMessageNotAllowedError,
    fetch_support_emails,
)
from backend.app.integrations.support_case_management import (
    get_support_case_detail,
    list_support_cases_dashboard,
    record_draft_ready,
    record_draft_saved,
    record_investigation_success,
    sync_support_cases,
    update_support_case_status,
)
from backend.app.integrations.support_draft import generate_support_draft
from backend.app.integrations.support_investigation import investigate_support_email
from backend.app.services.support_cases import MANUAL_STATUSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(title="AI Operations Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty")
        return value.strip()


class SaveGmailDraftRequest(BaseModel):
    draft_text: str = Field(min_length=1, max_length=10000)

    @field_validator("draft_text")
    @classmethod
    def validate_draft_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Draft text cannot be empty")
        return cleaned


class UpdateCaseStatusRequest(BaseModel):
    status: str = Field(min_length=1, max_length=64)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        cleaned = value.strip().casefold()
        if cleaned not in MANUAL_STATUSES:
            raise ValueError(
                "status must be one of: needs_attention, closed"
            )
        return cleaned


def _safe_record_case(action: str, callback) -> None:
    """Persist case lifecycle updates without failing the primary support action."""
    try:
        callback()
    except Exception:  # noqa: BLE001
        logger.exception("support_case_%s_failed", action)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/orders/{order_id}")
def read_order(order_id: int):
    order = get_order(order_id)

    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")

    return order


@app.get("/inventory/{product_name}")
def read_inventory(product_name: str):
    canonical = normalize_product_name(product_name)
    quantity = get_inventory(product_name)

    if canonical is None or quantity is None:
        raise HTTPException(status_code=404, detail="Product not found")

    return {"product": canonical, "quantity": quantity}


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    request_id = uuid4().hex[:8]
    logger.info("chat_started request_id=%s", request_id)

    try:
        result = agent_graph.invoke({
            "messages": [("user", request.message)]
        })

        answer = result["messages"][-1].content
        logger.info("chat_completed request_id=%s", request_id)
        return {"answer": answer}

    except Exception as exc:
        if not is_ai_service_error(exc):
            raise

        logger.exception(
            "AI service connection failed request_id=%s",
            request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable.",
        ) from exc


@app.get("/support/emails")
def list_support_emails():
    """Read-only localhost inbox view for labeled Gmail support messages."""
    request_id = uuid4().hex[:8]
    logger.info("support_emails_started request_id=%s", request_id)

    try:
        emails = fetch_support_emails()
    except GmailAuthError as exc:
        logger.warning("support_emails_auth_failed request_id=%s", request_id)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except GmailLabelNotFoundError as exc:
        logger.warning("support_emails_label_missing request_id=%s", request_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GmailApiError as exc:
        logger.warning("support_emails_gmail_failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "support_emails_completed request_id=%s count=%s",
        request_id,
        len(emails),
    )
    return {"emails": emails}


@app.post("/support/emails/{message_id}/investigate")
def investigate_support_email_endpoint(message_id: str):
    """Investigate one labeled support email with existing read-only tools."""
    request_id = uuid4().hex[:8]
    logger.info("support_investigate_started request_id=%s", request_id)

    try:
        findings = investigate_support_email(message_id)
    except GmailAuthError as exc:
        logger.warning("support_investigate_auth_failed request_id=%s", request_id)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except GmailLabelNotFoundError as exc:
        logger.warning("support_investigate_label_missing request_id=%s", request_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GmailMessageNotAllowedError as exc:
        logger.warning("support_investigate_not_allowed request_id=%s", request_id)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except GmailApiError as exc:
        logger.warning("support_investigate_gmail_failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        if is_ai_service_error(exc):
            logger.warning("support_investigate_ai_failed request_id=%s", request_id)
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable.",
            ) from exc
        raise

    logger.info(
        "support_investigate_completed request_id=%s intent=%s",
        request_id,
        findings.get("intent"),
    )
    _safe_record_case(
        "investigated",
        lambda: record_investigation_success(findings),
    )
    presentation = findings.get("presentation") or {}
    # Never return raw email body/sender from this endpoint.
    # Prefer operator presentation over raw tool dumps for UI consumption.
    return {
        "message_id": findings.get("message_id"),
        "intent": findings.get("intent"),
        "order_id": findings.get("order_id"),
        "olist_order_id": findings.get("olist_order_id"),
        "requested_topics": findings.get("requested_topics") or [],
        "verified_facts": findings.get("verified_facts") or "",
        "policy_information": findings.get("policy_information") or "",
        "uncertainties": findings.get("uncertainties") or [],
        "summary": presentation.get("summary") or findings.get("summary"),
        "presentation": {
            "data_scope": presentation.get("data_scope") or "live",
            "status": presentation.get("status") or "",
            "order_details": presentation.get("order_details") or "",
            "payment_details": presentation.get("payment_details") or "",
            "policy_information": presentation.get("policy_information")
            or findings.get("policy_information")
            or "",
            "uncertainties": presentation.get("uncertainties")
            or findings.get("uncertainties")
            or [],
            "summary": presentation.get("summary") or findings.get("summary") or "",
        },
    }


@app.post("/support/emails/{message_id}/draft")
def generate_support_draft_endpoint(message_id: str):
    """Generate a review-only customer reply draft from verified investigation facts."""
    request_id = uuid4().hex[:8]
    logger.info("support_draft_started request_id=%s", request_id)

    try:
        result = generate_support_draft(message_id)
    except GmailAuthError as exc:
        logger.warning("support_draft_auth_failed request_id=%s", request_id)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except GmailLabelNotFoundError as exc:
        logger.warning("support_draft_label_missing request_id=%s", request_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GmailMessageNotAllowedError as exc:
        logger.warning("support_draft_not_allowed request_id=%s", request_id)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except GmailApiError as exc:
        logger.warning("support_draft_gmail_failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        if is_ai_service_error(exc):
            logger.warning("support_draft_ai_failed request_id=%s", request_id)
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable.",
            ) from exc
        raise

    logger.info(
        "support_draft_completed request_id=%s used_fallback=%s",
        request_id,
        result.get("used_fallback"),
    )
    _safe_record_case(
        "draft_ready",
        lambda: record_draft_ready(result),
    )
    return {
        "message_id": result.get("message_id"),
        "intent": result.get("intent"),
        "order_id": result.get("order_id"),
        "olist_order_id": result.get("olist_order_id"),
        "draft": result.get("draft") or "",
        "used_fallback": bool(result.get("used_fallback")),
        "status": result.get("status") or "not_sent",
    }


@app.post("/support/emails/{message_id}/save-draft")
def save_support_gmail_draft_endpoint(
    message_id: str,
    request: SaveGmailDraftRequest,
):
    """Save the operator-edited reply as a Gmail draft (never send)."""
    request_id = uuid4().hex[:8]
    logger.info("support_save_draft_started request_id=%s", request_id)

    try:
        result = save_support_reply_draft(message_id, request.draft_text)
    except GmailDraftValidationError as exc:
        logger.warning("support_save_draft_invalid request_id=%s", request_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GmailAuthError as exc:
        logger.warning("support_save_draft_auth_failed request_id=%s", request_id)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except GmailLabelNotFoundError as exc:
        logger.warning("support_save_draft_label_missing request_id=%s", request_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GmailMessageNotAllowedError as exc:
        logger.warning("support_save_draft_not_allowed request_id=%s", request_id)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except GmailApiError as exc:
        logger.warning("support_save_draft_gmail_failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "support_save_draft_completed request_id=%s has_draft_id=%s",
        request_id,
        bool(result.get("draft_id")),
    )
    if result.get("draft_id"):
        _safe_record_case(
            "draft_saved",
            lambda: record_draft_saved(
                message_id=result.get("message_id") or message_id,
                draft_id=result.get("draft_id"),
                subject=result.get("source_subject") or None,
                thread_id=result.get("thread_id") or None,
            ),
        )
    return {
        "message_id": result.get("message_id"),
        "draft_id": result.get("draft_id"),
        "thread_id": result.get("thread_id"),
        "recipient": result.get("recipient"),
        "subject": result.get("subject"),
        "status": result.get("status") or "saved_not_sent",
        "sent": False,
    }


@app.post("/support/cases/sync")
def sync_support_cases_endpoint():
    """Sync labeled Gmail messages into persistent support cases (idempotent)."""
    request_id = uuid4().hex[:8]
    logger.info("support_cases_sync_started request_id=%s", request_id)

    try:
        result = sync_support_cases()
    except GmailAuthError as exc:
        logger.warning("support_cases_sync_auth_failed request_id=%s", request_id)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except GmailLabelNotFoundError as exc:
        logger.warning("support_cases_sync_label_missing request_id=%s", request_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GmailApiError as exc:
        logger.warning("support_cases_sync_gmail_failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("support_cases_sync_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=503,
            detail="Support case storage is unavailable.",
        ) from exc

    logger.info(
        "support_cases_sync_completed request_id=%s synced=%s",
        request_id,
        result.get("synced_count"),
    )
    return result


@app.get("/support/cases")
def list_support_cases_endpoint():
    """List persistent support cases with status counts."""
    request_id = uuid4().hex[:8]
    logger.info("support_cases_list_started request_id=%s", request_id)
    try:
        return list_support_cases_dashboard()
    except Exception as exc:
        logger.exception("support_cases_list_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=503,
            detail="Support case storage is unavailable.",
        ) from exc


@app.get("/support/cases/{case_id}")
def get_support_case_endpoint(case_id: int):
    """Return one case and its activity history."""
    request_id = uuid4().hex[:8]
    logger.info("support_case_detail_started request_id=%s", request_id)
    try:
        detail = get_support_case_detail(case_id)
    except Exception as exc:
        logger.exception("support_case_detail_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=503,
            detail="Support case storage is unavailable.",
        ) from exc

    if detail is None:
        raise HTTPException(status_code=404, detail="Support case not found.")
    return detail


@app.post("/support/cases/{case_id}/status")
def update_support_case_status_endpoint(
    case_id: int,
    request: UpdateCaseStatusRequest,
):
    """Manual operator status updates: needs_attention or closed only."""
    request_id = uuid4().hex[:8]
    logger.info(
        "support_case_status_started request_id=%s status=%s",
        request_id,
        request.status,
    )
    try:
        case = update_support_case_status(case_id, request.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("support_case_status_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=503,
            detail="Support case storage is unavailable.",
        ) from exc

    if case is None:
        raise HTTPException(status_code=404, detail="Support case not found.")
    return {"case": case}


# python -m uvicorn backend.app.main:app --reload

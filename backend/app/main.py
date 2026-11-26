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
from backend.app.integrations.support_draft import generate_support_draft
from backend.app.integrations.support_investigation import investigate_support_email

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
    return {
        "message_id": result.get("message_id"),
        "draft_id": result.get("draft_id"),
        "thread_id": result.get("thread_id"),
        "recipient": result.get("recipient"),
        "subject": result.get("subject"),
        "status": result.get("status") or "saved_not_sent",
        "sent": False,
    }


# python -m uvicorn backend.app.main:app --reload

import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.app.services.orders import get_order
from backend.app.services.inventory import get_inventory, normalize_product_name
from backend.app.agents.agent_graph import agent_graph
from backend.app.agents.ai_errors import is_ai_service_error
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(title="AI Operations Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["POST"],
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



# python -m uvicorn backend.app.main:app --reload

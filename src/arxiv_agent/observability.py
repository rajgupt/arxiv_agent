import os
import time
import httpx
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

_client: Langfuse | None = None


def get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_BASE_URL"],
        )
    return _client


def check_connection() -> bool:
    base_url = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
    try:
        r = httpx.get(f"{base_url}/api/public/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_trace(question: str):
    return get_client().trace(name="arxiv_agent_run", input=question)


def llm_span(trace, name: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, model: str):
    return trace.span(
        name=name,
        metadata={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        },
    )


def tool_span(
    trace,
    paper_id_or_query: str,
    fetch_latency_ms: float,
    chars_returned: int,
    success: bool,
    paper_title: str | None = None,
):
    return trace.span(
        name="arxiv_fetch",
        input=paper_id_or_query,
        metadata={
            "fetch_latency_ms": fetch_latency_ms,
            "chars_returned": chars_returned,
            "success": success,
            "paper_title": paper_title,
        },
    )


def flush():
    get_client().flush()

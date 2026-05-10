import os
import httpx
from langfuse import Langfuse
from langfuse._client.span import LangfuseSpan
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


def start_trace(question: str) -> LangfuseSpan:
    """Start a root agent span that acts as the trace for an agent run."""
    return get_client().start_observation(
        name="arxiv_agent_run",
        as_type="agent",
        input=question,
    )


def llm_span(trace: LangfuseSpan, name: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, model: str) -> LangfuseSpan:
    return trace.start_observation(
        name=name,
        as_type="generation",
        model=model,
        usage_details={
            "input": prompt_tokens,
            "output": completion_tokens,
        },
        metadata={"latency_ms": latency_ms},
    )


def tool_span(
    trace: LangfuseSpan,
    paper_id_or_query: str,
    fetch_latency_ms: float,
    chars_returned: int,
    success: bool,
    paper_title: str | None = None,
) -> LangfuseSpan:
    return trace.start_observation(
        name="arxiv_fetch",
        as_type="tool",
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

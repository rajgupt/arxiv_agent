# ArXiv Agent Observability & Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph ReAct agent that fetches arxiv papers to answer research questions, with manual Langfuse SDK instrumentation on every span and a structured eval framework scoring behavioral correctness and answer quality across 10+ test cases.

**Architecture:** LangGraph ReAct graph with `reason → tool → reason` nodes; each node manually creates Langfuse spans with timing and metadata. Eval runs as a separate notebook that loads saved run results, scores them with 4 functions, and posts scores back to Langfuse traces.

**Tech Stack:** Python 3.11+, uv, LangGraph, langchain-openai (OpenRouter), Langfuse SDK, httpx, feedparser, markdownify, pandas, jupyter

---

## File Map

| File | Role |
|---|---|
| `arxiv-agent/pyproject.toml` | uv project + all dependencies |
| `arxiv-agent/.env` | secrets (gitignored) |
| `arxiv-agent/.gitignore` | ignore `.env`, `data/run_results.json` |
| `arxiv-agent/src/arxiv_agent/__init__.py` | empty package marker |
| `arxiv-agent/src/arxiv_agent/observability.py` | Langfuse client singleton + `check_connection`, `start_trace`, `llm_span`, `tool_span` |
| `arxiv-agent/src/arxiv_agent/tools.py` | `arxiv_fetch(paper_id, search_query)` — returns markdown string |
| `arxiv-agent/src/arxiv_agent/agent.py` | LangGraph graph definition + `run_agent(question, trace)` |
| `arxiv-agent/src/arxiv_agent/eval.py` | `load_cases()`, `score_tool_called()`, `score_correct_paper()`, `score_keyword_coverage()`, `score_no_hallucination()` |
| `arxiv-agent/data/eval_cases.json` | 11 eval cases across 4 categories |
| `arxiv-agent/tests/test_tools.py` | Unit tests for `arxiv_fetch` |
| `arxiv-agent/tests/test_eval.py` | Unit tests for all 4 scoring functions |
| `arxiv-agent/notebooks/01_agent.ipynb` | Interactive agent notebook |
| `arxiv-agent/notebooks/02_eval.ipynb` | Eval runner + scoring + report notebook |

---

## Task 1: Project scaffold with uv

**Files:**
- Create: `arxiv-agent/pyproject.toml`
- Create: `arxiv-agent/.env`
- Create: `arxiv-agent/.gitignore`
- Create: `arxiv-agent/src/arxiv_agent/__init__.py`
- Create: `arxiv-agent/data/.gitkeep`
- Create: `arxiv-agent/tests/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd arxiv-agent
uv init --name arxiv-agent --python 3.11
```

- [ ] **Step 2: Add all dependencies**

```bash
uv add langgraph langchain-openai langfuse httpx python-dotenv pandas feedparser markdownify ipykernel jupyter
uv add --dev pytest
```

- [ ] **Step 3: Create src layout**

```bash
mkdir -p src/arxiv_agent tests data notebooks
touch src/arxiv_agent/__init__.py tests/__init__.py data/.gitkeep
```

- [ ] **Step 4: Create `.env`**

```
LANGFUSE_SECRET_KEY=sk-lf-02504799-2b91-46ce-a4d1-73befe1a3aa3
LANGFUSE_PUBLIC_KEY=pk-lf-df609cea-c4a1-4fce-9c25-cf1089d9266c
LANGFUSE_BASE_URL=http://localhost:3000
OPENROUTER_API_KEY=<your_key_here>
```

- [ ] **Step 5: Create `.gitignore`**

```
.env
data/run_results.json
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 6: Verify uv environment**

```bash
uv run python -c "import langgraph, langfuse, feedparser; print('ok')"
```

Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add arxiv-agent/
git commit -m "chore: scaffold arxiv-agent uv project"
```

---

## Task 2: Observability module

**Files:**
- Create: `arxiv-agent/src/arxiv_agent/observability.py`

- [ ] **Step 1: Write `observability.py`**

```python
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
```

- [ ] **Step 2: Verify import**

```bash
cd arxiv-agent
uv run python -c "from arxiv_agent.observability import check_connection; print(check_connection())"
```

Expected: `True` (Langfuse must be running at localhost:3000)

- [ ] **Step 3: Commit**

```bash
git add arxiv-agent/src/arxiv_agent/observability.py
git commit -m "feat: add Langfuse observability module"
```

---

## Task 3: arxiv_fetch tool

**Files:**
- Create: `arxiv-agent/src/arxiv_agent/tools.py`
- Create: `arxiv-agent/tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# arxiv-agent/tests/test_tools.py
import pytest
from unittest.mock import patch, MagicMock
from arxiv_agent.tools import arxiv_fetch


def test_fetch_by_paper_id_returns_markdown():
    result = arxiv_fetch(paper_id="1706.03762")
    assert isinstance(result, str)
    assert len(result) > 100
    assert "transformer" in result.lower() or "attention" in result.lower()


def test_fetch_by_search_query_returns_markdown():
    result = arxiv_fetch(search_query="attention is all you need transformer")
    assert isinstance(result, str)
    assert len(result) > 100


def test_fetch_bad_paper_id_returns_error_dict():
    result = arxiv_fetch(paper_id="9999.99999")
    assert isinstance(result, dict)
    assert "error" in result
    assert result["paper_id"] == "9999.99999"


def test_fetch_result_truncated_to_8000_chars():
    result = arxiv_fetch(paper_id="1706.03762")
    if isinstance(result, str):
        assert len(result) <= 8500  # small buffer over 8000


def test_fetch_requires_at_least_one_argument():
    with pytest.raises(ValueError, match="paper_id or search_query"):
        arxiv_fetch()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd arxiv-agent
uv run pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `tools.py` doesn't exist yet.

- [ ] **Step 3: Write `tools.py`**

```python
# arxiv-agent/src/arxiv_agent/tools.py
import httpx
import feedparser
from markdownify import markdownify

ARXIV_SEARCH_URL = "http://export.arxiv.org/api/query"
ARXIV_ABS_URL = "https://arxiv.org/abs/{}"
ARXIV_HTML_URL = "https://arxiv.org/html/{}"
MAX_CHARS = 8000


def arxiv_fetch(paper_id: str | None = None, search_query: str | None = None) -> str | dict:
    if not paper_id and not search_query:
        raise ValueError("Provide paper_id or search_query")

    if not paper_id and search_query:
        paper_id = _search_for_paper_id(search_query)
        if paper_id is None:
            return {"error": "No papers found for query", "query": search_query}

    return _fetch_paper_as_markdown(paper_id)


def _search_for_paper_id(query: str) -> str | None:
    params = {"search_query": f"all:{query}", "max_results": 1, "sortBy": "relevance"}
    try:
        r = httpx.get(ARXIV_SEARCH_URL, params=params, timeout=10)
        feed = feedparser.parse(r.text)
        if not feed.entries:
            return None
        entry_id = feed.entries[0].id  # e.g. http://arxiv.org/abs/1706.03762v5
        return entry_id.split("/abs/")[-1].split("v")[0]
    except Exception:
        return None


def _fetch_paper_as_markdown(paper_id: str) -> str | dict:
    # Try HTML version first for better markdown conversion
    for url in [ARXIV_HTML_URL.format(paper_id), ARXIV_ABS_URL.format(paper_id)]:
        try:
            r = httpx.get(url, timeout=15, follow_redirects=True)
            if r.status_code == 200:
                md = markdownify(r.text, heading_style="ATX")
                # Strip excessive whitespace
                lines = [l for l in md.splitlines() if l.strip()]
                clean = "\n".join(lines)
                return clean[:MAX_CHARS]
        except Exception:
            continue

    return {"error": f"Could not fetch paper", "paper_id": paper_id}
```

- [ ] **Step 4: Run tests**

```bash
cd arxiv-agent
uv run pytest tests/test_tools.py -v
```

Expected: `test_fetch_requires_at_least_one_argument` PASS. The live fetch tests require network — run them manually:

```bash
uv run pytest tests/test_tools.py -v -k "not bad_paper_id"
```

Expected: all pass (may take 5-10s for network calls).

- [ ] **Step 5: Commit**

```bash
git add arxiv-agent/src/arxiv_agent/tools.py arxiv-agent/tests/test_tools.py
git commit -m "feat: add arxiv_fetch tool with paper_id and search_query support"
```

---

## Task 4: LangGraph agent

**Files:**
- Create: `arxiv-agent/src/arxiv_agent/agent.py`

- [ ] **Step 1: Write `agent.py`**

```python
# arxiv-agent/src/arxiv_agent/agent.py
import os
import time
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from arxiv_agent.tools import arxiv_fetch as _arxiv_fetch
from arxiv_agent.observability import llm_span, tool_span
import json

SYSTEM_PROMPT = """You are a research assistant. For any research question, you MUST use the \
arxiv_fetch tool to retrieve the actual paper before answering. Never answer \
from training data alone. Always cite the paper ID you fetched.

If the question is clearly general (e.g. "what is gradient descent"), you may answer directly."""


@tool
def arxiv_fetch(paper_id: str = "", search_query: str = "") -> str:
    """Fetch an arxiv paper as markdown. Provide either paper_id (e.g. '1706.03762') or search_query."""
    result = _arxiv_fetch(
        paper_id=paper_id if paper_id else None,
        search_query=search_query if search_query else None,
    )
    if isinstance(result, dict):
        return json.dumps(result)
    return result


TOOLS = [arxiv_fetch]
TOOL_MAP = {t.name: t for t in TOOLS}

MODEL = "qwen/qwen3-35b-a3b"


def _make_llm():
    return ChatOpenAI(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ).bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    trace: object  # Langfuse trace — passed through but not serialized


def reason_node(state: AgentState) -> AgentState:
    llm = _make_llm()
    messages = state["messages"]
    trace = state.get("trace")

    start = time.time()
    response = llm.invoke(messages)
    latency_ms = (time.time() - start) * 1000

    usage = getattr(response, "usage_metadata", {}) or {}
    if trace:
        llm_span(
            trace=trace,
            name="llm_call",
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=round(latency_ms, 2),
            model=MODEL,
        )

    return {"messages": [response], "trace": trace}


def tool_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    trace = state.get("trace")
    results = []

    for tool_call in last.tool_calls:
        t = TOOL_MAP[tool_call["name"]]
        args = tool_call["args"]

        start = time.time()
        result = t.invoke(args)
        latency_ms = (time.time() - start) * 1000

        if trace:
            query = args.get("paper_id") or args.get("search_query") or ""
            success = not (isinstance(result, str) and '"error"' in result)
            tool_span(
                trace=trace,
                paper_id_or_query=query,
                fetch_latency_ms=round(latency_ms, 2),
                chars_returned=len(result),
                success=success,
            )

        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    return {"messages": results, "trace": trace}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("reason", reason_node)
    g.add_node("tool", tool_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", should_continue, {"tool": "tool", END: END})
    g.add_edge("tool", "reason")
    return g.compile()


def run_agent(question: str, trace=None) -> dict:
    """Run agent for one question. Returns {answer, trace_id, messages}."""
    graph = build_graph()
    init_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    result = graph.invoke({"messages": init_messages, "trace": trace})
    final_answer = result["messages"][-1].content
    if trace:
        trace.update(output=final_answer)
    return {
        "answer": final_answer,
        "trace_id": trace.id if trace else None,
        "messages": result["messages"],
    }
```

- [ ] **Step 2: Verify import**

```bash
cd arxiv-agent
uv run python -c "from arxiv_agent.agent import build_graph; g = build_graph(); print('graph ok')"
```

Expected: `graph ok`

- [ ] **Step 3: Commit**

```bash
git add arxiv-agent/src/arxiv_agent/agent.py
git commit -m "feat: add LangGraph ReAct agent with Langfuse span instrumentation"
```

---

## Task 5: Eval dataset

**Files:**
- Create: `arxiv-agent/data/eval_cases.json`

- [ ] **Step 1: Write `eval_cases.json`**

```json
[
  {
    "id": "kp_01",
    "category": "known_paper",
    "question": "What is the main contribution of the Attention Is All You Need paper?",
    "expected_paper_id": "1706.03762",
    "expected_keywords": ["self-attention", "transformer", "encoder", "decoder"],
    "expects_tool_call": true
  },
  {
    "id": "kp_02",
    "category": "known_paper",
    "question": "Summarize the RLHF paper by Christiano et al. 2017",
    "expected_paper_id": "1706.03741",
    "expected_keywords": ["human feedback", "reward model", "preference"],
    "expects_tool_call": true
  },
  {
    "id": "kp_03",
    "category": "known_paper",
    "question": "What does the BERT paper propose?",
    "expected_paper_id": "1810.04805",
    "expected_keywords": ["bidirectional", "pre-training", "fine-tuning", "masked"],
    "expects_tool_call": true
  },
  {
    "id": "kp_04",
    "category": "known_paper",
    "question": "Explain the LoRA fine-tuning method from paper 2106.09685",
    "expected_paper_id": "2106.09685",
    "expected_keywords": ["low-rank", "adaptation", "weight matrix", "parameters"],
    "expects_tool_call": true
  },
  {
    "id": "sq_01",
    "category": "search_query",
    "question": "Find a paper on retrieval augmented generation and explain how it works",
    "expected_paper_id": null,
    "expected_keywords": ["retrieval", "generation", "knowledge", "document"],
    "expects_tool_call": true
  },
  {
    "id": "sq_02",
    "category": "search_query",
    "question": "What recent work exists on chain of thought prompting?",
    "expected_paper_id": null,
    "expected_keywords": ["chain of thought", "reasoning", "prompting", "step"],
    "expects_tool_call": true
  },
  {
    "id": "sq_03",
    "category": "search_query",
    "question": "Find papers on mixture of experts models",
    "expected_paper_id": null,
    "expected_keywords": ["expert", "routing", "sparse", "mixture"],
    "expects_tool_call": true
  },
  {
    "id": "nt_01",
    "category": "no_tool_needed",
    "question": "What is the difference between supervised and unsupervised learning?",
    "expected_paper_id": null,
    "expected_keywords": ["labeled", "unlabeled", "classification", "clustering"],
    "expects_tool_call": false
  },
  {
    "id": "nt_02",
    "category": "no_tool_needed",
    "question": "What does gradient descent do in neural network training?",
    "expected_paper_id": null,
    "expected_keywords": ["gradient", "loss", "optimization", "weights"],
    "expects_tool_call": false
  },
  {
    "id": "bi_01",
    "category": "bad_input",
    "question": "Tell me about arxiv paper 9999.99999",
    "expected_paper_id": "9999.99999",
    "expected_keywords": [],
    "expects_tool_call": true
  },
  {
    "id": "bi_02",
    "category": "bad_input",
    "question": "asdkjhasd lkjhaksjdh paper please",
    "expected_paper_id": null,
    "expected_keywords": [],
    "expects_tool_call": false
  }
]
```

- [ ] **Step 2: Validate JSON parses**

```bash
cd arxiv-agent
uv run python -c "import json; cases = json.load(open('data/eval_cases.json')); print(f'{len(cases)} cases loaded')"
```

Expected: `11 cases loaded`

- [ ] **Step 3: Commit**

```bash
git add arxiv-agent/data/eval_cases.json
git commit -m "feat: add eval dataset with 11 cases across 4 categories"
```

---

## Task 6: Eval scoring functions

**Files:**
- Create: `arxiv-agent/src/arxiv_agent/eval.py`
- Create: `arxiv-agent/tests/test_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# arxiv-agent/tests/test_eval.py
import pytest
from arxiv_agent.eval import (
    score_tool_called,
    score_correct_paper,
    score_keyword_coverage,
    score_no_hallucination,
    load_cases,
)


def _make_run(tool_calls=None, answer=""):
    return {
        "answer": answer,
        "tool_calls_made": tool_calls or [],
    }


def test_score_tool_called_when_tool_was_used():
    run = _make_run(tool_calls=[{"name": "arxiv_fetch", "args": {"paper_id": "1706.03762"}}])
    assert score_tool_called(run, expects_tool_call=True) == 1


def test_score_tool_called_when_tool_not_used_but_expected():
    run = _make_run(tool_calls=[])
    assert score_tool_called(run, expects_tool_call=True) == 0


def test_score_tool_called_correct_when_no_tool_expected_and_none_used():
    run = _make_run(tool_calls=[])
    assert score_tool_called(run, expects_tool_call=False) == 1


def test_score_correct_paper_match():
    run = _make_run(tool_calls=[{"name": "arxiv_fetch", "args": {"paper_id": "1706.03762"}}])
    assert score_correct_paper(run, expected_paper_id="1706.03762") == 1


def test_score_correct_paper_mismatch():
    run = _make_run(tool_calls=[{"name": "arxiv_fetch", "args": {"paper_id": "1810.04805"}}])
    assert score_correct_paper(run, expected_paper_id="1706.03762") == 0


def test_score_correct_paper_skipped_when_no_expected():
    run = _make_run(tool_calls=[])
    assert score_correct_paper(run, expected_paper_id=None) is None


def test_score_keyword_coverage_full():
    run = _make_run(answer="The transformer uses self-attention, encoder, decoder architecture.")
    assert score_keyword_coverage(run, ["self-attention", "encoder", "decoder"]) == 1.0


def test_score_keyword_coverage_partial():
    run = _make_run(answer="The transformer uses self-attention.")
    score = score_keyword_coverage(run, ["self-attention", "encoder", "decoder"])
    assert abs(score - 1/3) < 0.01


def test_score_keyword_coverage_empty_keywords():
    run = _make_run(answer="anything")
    assert score_keyword_coverage(run, []) == 1.0


def test_score_no_hallucination_passes_when_no_tool_and_not_expected():
    run = _make_run(tool_calls=[])
    assert score_no_hallucination(run, expects_tool_call=False) == 1


def test_score_no_hallucination_fails_when_tool_used_but_not_expected():
    run = _make_run(tool_calls=[{"name": "arxiv_fetch", "args": {}}])
    assert score_no_hallucination(run, expects_tool_call=False) == 0


def test_load_cases_returns_list():
    cases = load_cases()
    assert isinstance(cases, list)
    assert len(cases) == 11
    assert all("id" in c for c in cases)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd arxiv-agent
uv run pytest tests/test_eval.py -v
```

Expected: `ImportError` — `eval.py` doesn't exist yet.

- [ ] **Step 3: Write `eval.py`**

```python
# arxiv-agent/src/arxiv_agent/eval.py
import json
from pathlib import Path


def load_cases(path: str = "data/eval_cases.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def score_tool_called(run: dict, expects_tool_call: bool) -> int:
    """1 if tool-call expectation matches reality, else 0."""
    tool_was_called = len(run.get("tool_calls_made", [])) > 0
    return int(tool_was_called == expects_tool_call)


def score_correct_paper(run: dict, expected_paper_id: str | None) -> int | None:
    """1 if fetched paper_id matches expected. None if expected_paper_id is null (skip)."""
    if expected_paper_id is None:
        return None
    for tc in run.get("tool_calls_made", []):
        if tc.get("args", {}).get("paper_id") == expected_paper_id:
            return 1
    return 0


def score_keyword_coverage(run: dict, expected_keywords: list[str]) -> float:
    """Fraction of expected_keywords present in answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer = run.get("answer", "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer)
    return hits / len(expected_keywords)


def score_no_hallucination(run: dict, expects_tool_call: bool) -> int:
    """For no_tool_needed cases: 1 if agent did NOT call tool. Skip for cases that expect a tool call."""
    if expects_tool_call:
        return 1  # not applicable, pass through
    tool_was_called = len(run.get("tool_calls_made", [])) > 0
    return int(not tool_was_called)
```

- [ ] **Step 4: Run tests**

```bash
cd arxiv-agent
uv run pytest tests/test_eval.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add arxiv-agent/src/arxiv_agent/eval.py arxiv-agent/tests/test_eval.py
git commit -m "feat: add eval scoring functions with full test coverage"
```

---

## Task 7: Notebook 01 — Interactive agent

**Files:**
- Create: `arxiv-agent/notebooks/01_agent.ipynb`

- [ ] **Step 1: Create the notebook**

Create `arxiv-agent/notebooks/01_agent.ipynb` with the following cells (use `jupyter notebook` or paste JSON directly):

**Cell 1 — Setup & connectivity check:**
```python
import sys, os
sys.path.insert(0, "../src")

from dotenv import load_dotenv
load_dotenv("../.env")

from arxiv_agent.observability import check_connection, get_client, flush

assert check_connection(), "Langfuse not reachable at localhost:3000 — is it running?"
print("Langfuse connection OK")
print(f"Langfuse UI: {os.environ['LANGFUSE_BASE_URL']}")
```

**Cell 2 — Single interactive query:**
```python
from arxiv_agent.observability import start_trace, flush
from arxiv_agent.agent import run_agent

question = "What is the main contribution of the Attention Is All You Need paper?"

trace = start_trace(question)
result = run_agent(question, trace=trace)
flush()

print(f"\nAnswer:\n{result['answer']}")
print(f"\nLangfuse trace: {os.environ['LANGFUSE_BASE_URL']}/trace/{result['trace_id']}")
```

**Cell 3 — Inspect raw trace:**
```python
from langfuse import Langfuse

lf = get_client()
# Give Langfuse a moment to process
import time; time.sleep(1)

raw = lf.get_trace(result["trace_id"])
print(f"Trace ID: {raw.id}")
print(f"Input: {raw.input}")
print(f"Output: {raw.output}")
print(f"\nSpans ({len(raw.observations)}):")
for obs in raw.observations:
    print(f"  [{obs.type}] {obs.name} — {obs.metadata}")
```

**Cell 4 — Batch run all eval cases:**
```python
import json
from pathlib import Path

cases = json.loads(Path("../data/eval_cases.json").read_text())
results = []

for case in cases:
    print(f"Running {case['id']}: {case['question'][:60]}...")
    trace = start_trace(case["question"])
    
    # Extract tool calls from messages
    from arxiv_agent.agent import build_graph
    from langchain_core.messages import HumanMessage, SystemMessage
    from arxiv_agent.agent import SYSTEM_PROMPT
    
    graph = build_graph()
    msgs = graph.invoke({
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=case["question"])],
        "trace": trace,
    })
    
    final_answer = msgs["messages"][-1].content
    trace.update(output=final_answer)
    
    tool_calls_made = [
        {"name": m.name if hasattr(m, "name") else "", "args": {}}
        for m in msgs["messages"]
        if hasattr(m, "type") and m.type == "tool"
    ]
    # Capture tool call args from AIMessage tool_calls
    tool_calls_made = []
    for m in msgs["messages"]:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls_made.append({"name": tc["name"], "args": tc["args"]})
    
    results.append({
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": final_answer,
        "trace_id": trace.id,
        "tool_calls_made": tool_calls_made,
    })
    flush()

Path("../data/run_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved {len(results)} results to data/run_results.json")
```

- [ ] **Step 2: Run Cell 1 manually in Jupyter to verify Langfuse connects**

```bash
cd arxiv-agent
uv run jupyter notebook notebooks/01_agent.ipynb
```

Run Cell 1. Expected output: `Langfuse connection OK`

- [ ] **Step 3: Run Cell 2 to send first trace**

Run Cell 2. Expected: answer printed + Langfuse URL printed. Open the URL — you should see one trace with 2 spans (llm_call, arxiv_fetch).

- [ ] **Step 4: Commit**

```bash
git add arxiv-agent/notebooks/01_agent.ipynb
git commit -m "feat: add 01_agent notebook with Langfuse connectivity check and batch runner"
```

---

## Task 8: Notebook 02 — Eval runner

**Files:**
- Create: `arxiv-agent/notebooks/02_eval.ipynb`

- [ ] **Step 1: Create the notebook**

Create `arxiv-agent/notebooks/02_eval.ipynb` with the following cells:

**Cell 1 — Load data:**
```python
import sys, json, os
sys.path.insert(0, "../src")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("../.env")

cases = json.loads(Path("../data/eval_cases.json").read_text())
results = json.loads(Path("../data/run_results.json").read_text())

# Index by id for lookup
case_map = {c["id"]: c for c in cases}
print(f"Loaded {len(cases)} cases, {len(results)} run results")
```

**Cell 2 — Score all results:**
```python
import pandas as pd
from arxiv_agent.eval import (
    score_tool_called,
    score_correct_paper,
    score_keyword_coverage,
    score_no_hallucination,
)

rows = []
for run in results:
    case = case_map[run["id"]]
    row = {
        "id": run["id"],
        "category": case["category"],
        "question": case["question"][:60],
        "trace_id": run["trace_id"],
        "tool_called": score_tool_called(run, case["expects_tool_call"]),
        "correct_paper": score_correct_paper(run, case.get("expected_paper_id")),
        "keyword_coverage": score_keyword_coverage(run, case.get("expected_keywords", [])),
        "no_hallucination": score_no_hallucination(run, case["expects_tool_call"]),
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(df[["id", "category", "tool_called", "correct_paper", "keyword_coverage", "no_hallucination"]])
```

**Cell 3 — Post scores to Langfuse:**
```python
from arxiv_agent.observability import get_client, flush

lf = get_client()
score_cols = ["tool_called", "correct_paper", "keyword_coverage", "no_hallucination"]

for _, row in df.iterrows():
    for score_name in score_cols:
        val = row[score_name]
        if val is None:
            continue
        lf.score(
            trace_id=row["trace_id"],
            name=score_name,
            value=float(val),
        )

flush()
print("Scores posted to Langfuse. Open the UI to see them on each trace.")
print(f"UI: {os.environ['LANGFUSE_BASE_URL']}")
```

**Cell 4 — Summary report:**
```python
print("=== Overall Scores ===")
for col in score_cols:
    valid = df[col].dropna()
    print(f"  {col}: {valid.mean():.2f} ({valid.sum():.0f}/{len(valid)} pass)")

print("\n=== By Category ===")
for cat in df["category"].unique():
    sub = df[df["category"] == cat]
    print(f"\n  {cat} ({len(sub)} cases):")
    for col in score_cols:
        valid = sub[col].dropna()
        if len(valid) > 0:
            print(f"    {col}: {valid.mean():.2f}")
```

**Cell 5 — Failure analysis:**
```python
base_url = os.environ["LANGFUSE_BASE_URL"]

print("=== Failing Cases ===\n")
for _, row in df.iterrows():
    failures = []
    for col in score_cols:
        if row[col] is not None and row[col] < 1.0:
            failures.append(f"{col}={row[col]:.2f}")
    if failures:
        print(f"[{row['id']}] {row['question']}")
        print(f"  Failed: {', '.join(failures)}")
        print(f"  Trace: {base_url}/trace/{row['trace_id']}")
        print()
```

- [ ] **Step 2: Run 02_eval.ipynb after 01_agent.ipynb has produced run_results.json**

```bash
cd arxiv-agent
uv run jupyter notebook notebooks/02_eval.ipynb
```

Run all cells top to bottom. Expected: DataFrame printed, scores posted, summary report, failure trace URLs.

- [ ] **Step 3: Verify scores appear in Langfuse UI**

Open `http://localhost:3000`, navigate to Traces, click any trace — scores should appear in the Scores panel on the right.

- [ ] **Step 4: Commit**

```bash
git add arxiv-agent/notebooks/02_eval.ipynb
git commit -m "feat: add 02_eval notebook with scoring, Langfuse score posting, and failure analysis"
```

---

## Task 9: Final wiring check

- [ ] **Step 1: Run all unit tests**

```bash
cd arxiv-agent
uv run pytest tests/ -v
```

Expected: all unit tests pass (tool live-network tests are marked integration, skip if offline).

- [ ] **Step 2: End-to-end smoke test**

Run `01_agent.ipynb` Cell 1–3 (single query). Confirm:
- Answer printed in notebook
- Trace URL printed
- Opening URL shows trace with ≥2 spans (one `llm_call`, one `arxiv_fetch`)
- Spans have `latency_ms`, `prompt_tokens`, `completion_tokens` in metadata

- [ ] **Step 3: Full eval run**

Run `01_agent.ipynb` Cell 4 (all 11 cases). Then run all cells of `02_eval.ipynb`. Confirm:
- `data/run_results.json` written with 11 entries
- DataFrame shows all 4 scores per case
- Langfuse UI shows scores on each trace

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final wiring — all tasks complete"
```

---

## The Learning Loop (reference)

```
Edit system prompt in agent.py
  ↓
Re-run 01_agent.ipynb Cell 4     ← new run_results.json
  ↓
Re-run 02_eval.ipynb             ← new scores posted to Langfuse
  ↓
Open Langfuse UI → compare trace scores between runs
  ↓
Click a failing trace → see which span failed and why
  ↓
Diagnose and iterate
```

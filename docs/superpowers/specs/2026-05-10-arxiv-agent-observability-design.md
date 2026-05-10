# ArXiv Research Agent — Observability & Eval Design

**Date:** 2026-05-10  
**Goal:** Build a LangGraph ReAct agent that fetches arxiv papers to answer research questions, with full Langfuse observability (manual SDK instrumentation) and a structured eval framework that scores behavioral correctness and answer quality.

**Primary learning objective:** Understand how observability works by instrumenting every span manually — not via auto-callback — so the learner sees exactly what gets captured and why.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Agent framework | LangGraph (ReAct) | Explicit graph = explicit spans |
| LLM | `qwen/qwen3-35b-a3b` via OpenRouter | As specified |
| Observability | Langfuse SDK (manual) | Forces understanding of trace/span model |
| Package manager | uv | Fast, modern |
| Interface | Jupyter notebooks | Step-by-step learning, correlate output with UI |

---

## Project Structure

```
arxiv-agent/
├── pyproject.toml              # uv project + dependencies
├── .env                        # LANGFUSE_*, OPENROUTER_API_KEY (gitignored)
├── notebooks/
│   ├── 01_agent.ipynb          # Agent definition + interactive Q&A
│   └── 02_eval.ipynb           # Eval runner + scoring + report
├── src/
│   └── arxiv_agent/
│       ├── agent.py            # LangGraph ReAct graph definition
│       ├── tools.py            # arxiv_fetch tool (returns markdown)
│       ├── observability.py    # Langfuse client + span helpers
│       └── eval.py             # Eval dataset loader + scoring functions
└── data/
    └── eval_cases.json         # 10+ test cases with ground truth
```

---

## Architecture & Data Flow

```
User question (notebook cell)
  │
  ▼
langfuse.trace() started        ← trace_id, input question, session metadata
  │
  ▼
LangGraph agent.invoke()
  │
  ├─► [reason node]
  │     langfuse.span("llm_call")
  │       - model name
  │       - prompt_tokens, completion_tokens
  │       - latency_ms
  │       - decision: tool_call or final_answer
  │
  ├─► [tool node]  (if agent decided to fetch)
  │     langfuse.span("arxiv_fetch")
  │       - input: paper_id or search_query
  │       - fetch_latency_ms
  │       - chars_returned
  │       - success: bool
  │       - paper_title (if found)
  │
  └─► [reason node] (second pass with paper content)
        langfuse.span("llm_call")
          - prompt_tokens, completion_tokens
          - latency_ms
          - final answer generated
  │
  ▼
trace.update(output=final_answer)
langfuse.flush()                ← REQUIRED: sends buffered events to localhost:3000
  │
  ▼
Trace visible in Langfuse UI    ← answer is in notebook, trace is in Langfuse
```

**Key concept:** The final answer lives in the notebook output. Langfuse stores the *how* — every LLM call, every tool call, every intermediate step, with timing and token metadata.

---

## Agent Design

### Graph (LangGraph ReAct)

```
[START] → reason → tool → reason → [END]
           ↑_______________|
           (loops if more tool calls needed)
```

Conditional edge after `reason`: if LLM output contains a tool call → route to `tool`, else → `END`.

### System prompt

```
You are a research assistant. For any research question, you MUST use the 
arxiv_fetch tool to retrieve the actual paper before answering. Never answer 
from training data alone. Always cite the paper ID you fetched.
```

This prompt is intentionally strict to make the `tool_called` eval signal meaningful.

### `arxiv_fetch` tool

- **Input A:** `paper_id: str` (e.g. `"1706.03762"`) → fetch `https://arxiv.org/abs/{paper_id}`
- **Input B:** `search_query: str` (e.g. `"attention transformer 2017"`) → call arxiv API, pick top result, fetch that paper
- **Output:** Markdown string with title, authors, abstract, and key sections, truncated to ~8000 chars
- **On failure:** Return `{"error": "...", "paper_id": "..."}` — do not raise, let agent handle gracefully

### LLM client

OpenRouter via OpenAI-compatible API:
```
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-v4-flash"
api_key = OPENROUTER_API_KEY
```

---

## Observability Design

### Langfuse concepts used

| Concept | What it maps to in this agent |
|---|---|
| `trace` | One full agent run (question → answer) |
| `span` | One node execution (LLM call, tool call) |
| `score` | Eval result posted back to a trace |
| `flush()` | Send buffered events — always call after agent.invoke() |

### `observability.py`

```python
# Singleton Langfuse client
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_BASE_URL,  # http://localhost:3000
)

def check_langfuse_connection() -> bool:
    """Ping /api/public/health before running agent."""
    ...

def start_trace(question: str) -> trace:
    """Start a new trace for one agent run."""
    ...

def llm_span(trace, name, prompt_tokens, completion_tokens, latency_ms, model):
    """Record one LLM call as a span on the trace."""
    ...

def tool_span(trace, paper_id_or_query, fetch_latency_ms, chars_returned, success, paper_title):
    """Record one tool call as a span on the trace."""
    ...
```

### Connectivity check (Cell 1 of `01_agent.ipynb`)

Before any agent run, assert Langfuse is reachable:
```python
assert check_langfuse_connection(), "Langfuse not reachable at localhost:3000 — is it running?"
```

---

## Eval Dataset

**`data/eval_cases.json`** — 10+ cases across 4 categories:

### Categories

| Category | Count | Purpose |
|---|---|---|
| `known_paper` | 4 | Specific paper, known ID, known keywords — tests full pipeline |
| `search_query` | 3 | No paper ID given — agent must search by topic |
| `no_tool_needed` | 2 | General ML question — agent should NOT fetch |
| `bad_input` | 2 | Nonexistent ID / garbage query — tests graceful failure |

### Case schema

```json
{
  "id": "kp_01",
  "category": "known_paper",
  "question": "What is the main contribution of the Attention Is All You Need paper?",
  "expected_paper_id": "1706.03762",
  "expected_keywords": ["self-attention", "transformer", "encoder", "decoder"],
  "expects_tool_call": true
}
```

### Sample cases

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
    "question": "Explain the LoRA fine-tuning method",
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

---

## Scoring Functions

Run in `02_eval.ipynb`, scores posted to Langfuse via `langfuse.score()`.

| Score name | Type | Logic |
|---|---|---|
| `tool_called` | binary (0/1) | Did agent invoke `arxiv_fetch`? Checked from trace spans |
| `correct_paper` | binary (0/1) | Does fetched `paper_id` match `expected_paper_id`? (skip if null) |
| `keyword_coverage` | float 0.0–1.0 | Fraction of `expected_keywords` found in final answer (case-insensitive) |
| `no_hallucination` | binary (0/1) | For `no_tool_needed`: did agent answer WITHOUT calling fetch? |

Scores post to trace:
```python
langfuse.score(trace_id=trace_id, name="tool_called", value=1)
langfuse.score(trace_id=trace_id, name="keyword_coverage", value=0.75)
```

---

## Notebook Design

### `01_agent.ipynb`

| Cell | Purpose |
|---|---|
| 1 | Setup: load `.env`, init Langfuse, **connectivity check**, build agent |
| 2 | Single interactive query — print answer + Langfuse trace URL |
| 3 | Inspect raw trace dict inline (spans, latencies, token counts) |
| 4 | Batch run all eval cases → save `data/run_results.json` |

### `02_eval.ipynb`

| Cell | Purpose |
|---|---|
| 1 | Load `run_results.json` + `eval_cases.json` |
| 2 | Run all 4 scoring functions, build results DataFrame |
| 3 | Post scores to Langfuse (`langfuse.score()` for each trace) |
| 4 | Summary report: pass/fail per category, overall score |
| 5 | Failure analysis: for each failing case, print Langfuse trace URL → click to see which span failed |

---

## The Learning Loop

```
Run 01_agent (Cell 4) → run_results.json
  ↓
Run 02_eval → scores in DataFrame + posted to Langfuse
  ↓
Open Langfuse UI → find failing trace → inspect spans
  ↓
Diagnose: wrong tool call? bad parse? LLM ignored paper?
  ↓
Fix: adjust system prompt / tool logic / truncation limit
  ↓
Re-run → scores improve
```

This loop is the core of eval-driven improvement.

---

## Dependencies

```toml
[project]
name = "arxiv-agent"
requires-python = ">=3.11"
dependencies = [
    "langgraph",
    "langchain-openai",
    "langfuse",
    "httpx",
    "python-dotenv",
    "jupyter",
    "pandas",
    "feedparser",        # arxiv API response parsing
    "markdownify",       # HTML → markdown for paper content
]
```

---

## Key Constraints

- `langfuse.flush()` must be called after every `agent.invoke()` — without it, traces stay buffered and don't appear in UI
- Langfuse connectivity check runs before any agent execution
- Paper content truncated to ~8000 chars to stay within model context limits
- `.env` is gitignored — credentials never committed
- `no_tool_needed` cases: system prompt allows skipping fetch for clearly general questions, but agent must justify in output

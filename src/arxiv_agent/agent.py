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

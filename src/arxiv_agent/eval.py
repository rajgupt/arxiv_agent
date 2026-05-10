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

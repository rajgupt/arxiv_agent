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

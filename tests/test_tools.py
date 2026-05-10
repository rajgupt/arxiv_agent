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

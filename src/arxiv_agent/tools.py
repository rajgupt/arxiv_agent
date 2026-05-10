import httpx
import arxiv
from markdownify import markdownify

ARXIV_ABS_URL = "https://arxiv.org/abs/{}"
ARXIV_HTML_URL = "https://arxiv.org/html/{}"
MAX_CHARS = 8000

_arxiv_client = arxiv.Client()


def arxiv_fetch(paper_id: str | None = None, search_query: str | None = None) -> str | dict:
    if not paper_id and not search_query:
        raise ValueError("Provide paper_id or search_query")

    if not paper_id and search_query:
        paper_id = _search_for_paper_id(search_query)
        if paper_id is None:
            return {"error": "No papers found for query", "query": search_query}

    return _fetch_paper_as_markdown(paper_id)


def _search_for_paper_id(query: str) -> str | None:
    try:
        results = list(_arxiv_client.results(arxiv.Search(query=query, max_results=1)))
        if results:
            return results[0].entry_id.split("/abs/")[-1].split("v")[0]
    except Exception:
        pass
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

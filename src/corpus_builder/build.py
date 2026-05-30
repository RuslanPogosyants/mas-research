"""Build the recommender corpus from the Semantic Scholar Graph API (no key).

Run manually before the live F6 demo:
    python -m src.corpus_builder.build
Fetches paper metadata + abstracts for a set of queries, embeds the abstracts
with the configured model, and writes corpus/papers.jsonl + corpus/papers.npy.
Network + ML deps required; NOT part of CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_QUERIES = [
    "machine learning",
    "natural language processing",
    "algorithms data structures",
    "information retrieval",
    "multi-agent systems",
]
_FIELDS = "title,abstract,year,url,authors"
_PER_QUERY = 100


def fetch_papers(queries: list[str] = _QUERIES, per_query: int = _PER_QUERY) -> list[dict[str, Any]]:
    """Fetch deduplicated papers (with abstracts) for the given queries."""
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for query in queries:
            response = client.get(_API, params={"query": query, "limit": per_query, "fields": _FIELDS})
            response.raise_for_status()
            for item in response.json().get("data", []):
                abstract = item.get("abstract")
                title = item.get("title")
                if not abstract or not title or title in seen:
                    continue
                seen.add(title)
                authors = ", ".join(author.get("name", "") for author in item.get("authors", []))
                papers.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "authors": authors or None,
                        "year": item.get("year"),
                        "url": item.get("url"),
                    }
                )
    return papers


def build(out_dir: str = "corpus", model: str = "intfloat/multilingual-e5-base") -> int:
    """Fetch, embed, and persist the corpus. Returns the number of papers written."""
    import numpy
    from sentence_transformers import SentenceTransformer

    papers = fetch_papers()
    if not papers:
        return 0
    encoder = SentenceTransformer(model)
    embeddings = encoder.encode([f"passage: {paper['abstract']}" for paper in papers], normalize_embeddings=True)
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "papers.jsonl").open("w", encoding="utf-8") as handle:
        for paper in papers:
            meta = {key: paper[key] for key in ("title", "authors", "year", "url")}
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
    numpy.save(base / "papers.npy", embeddings)
    return len(papers)


if __name__ == "__main__":
    count = build()
    print(f"wrote {count} papers to corpus/")

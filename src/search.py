"""
Query the index from the command line. No LLM, no API key, no network.

This exists to separate two failures that look identical from the outside.
When a RAG system answers badly, the cause is either that retrieval returned
the wrong passages or that generation mishandled the right ones. Reading the
retrieved chunks directly settles it in seconds, and settles it without
spending tokens or waiting on a provider.

It is also the honest first demo: if the passages coming back here are not
the ones a human would have picked, no prompt will rescue the answer.

    python -m src.search "cholera cases in Banadir"
    python -m src.search "displacement figures" --month 2025-03
    python -m src.search "funding shortfall" --sector Funding --source OCHA
    python -m src.search "AWD outbreak" --region Hiraan --from 2022 --to 2024 -k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

from .chunker import tidy
from .embed import ChunkIndex
from .ingest import load_config, quiet_logs
from .retrieve import Filters, Retriever, infer_filters, load_parents

log = logging.getLogger("search")


def build_retriever(cfg: dict, embedder=None) -> Retriever:
    processed = Path(cfg["paths"]["processed"])
    index_dir = processed / "index"
    if not index_dir.exists():
        raise SystemExit(
            f"No index at {index_dir}. Build it first:\n"
            "    python -m src.chunker\n"
            "    python -m src.embed"
        )
    index = ChunkIndex.load(index_dir)
    if embedder is None:
        from .embed import SentenceTransformerEmbedder

        # The index records which model built it. Querying with a different
        # model returns confident nonsense — the vectors are the same shape
        # and the cosines are meaningless.
        embedder = SentenceTransformerEmbedder(index.model_name)
    return Retriever(index, embedder, parents=load_parents(processed / "parents.jsonl"))


def format_hit(n: int, hit, width: int = 96, snippet: int = 400) -> str:
    date = (hit.date_original or "undated")[:10]
    src = "/".join(hit.sources or []) or "?"
    facets = " | ".join(filter(None, [
        hit.sector, "/".join(getattr(hit, "regions", []) or []) or None,
    ]))
    head = f"[{n}] {date}  {src}" + (f"  ({facets})" if facets else "")
    body = textwrap.fill(" ".join(tidy(hit.body).split())[:snippet], width=width,
                         initial_indent="    ", subsequent_indent="    ")
    scores = f"    cosine {hit.dense_score:.3f}"
    if hit.rank_dense:
        scores += f"  dense#{hit.rank_dense}"
    if hit.rank_lexical:
        scores += f"  bm25#{hit.rank_lexical}"
    return f"{head}\n{body}\n{scores}\n    {hit.url or ''}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.search",
                                 description="Search the Somalia sitrep index.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--config", default="config/corpus.yaml")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--month", help="exact month, e.g. 2025-03 (repeatable)",
                    action="append")
    ap.add_argument("--from", dest="date_from", help="YYYY, YYYY-MM or YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--sector", action="append")
    ap.add_argument("--region", action="append")
    ap.add_argument("--source", action="append")
    ap.add_argument("--figures-only", action="store_true",
                    help="only chunks containing numbers")
    ap.add_argument("--per-parent", type=int, default=1,
                    help="max chunks from one section (default 1; consecutive "
                         "chunks of a section overlap by construction)")
    ap.add_argument("--per-report", type=int, default=2,
                    help="max chunks from any one report (default 2)")
    ap.add_argument("--spread", action="store_true",
                    help="at most one hit per reporting period — use to see how "
                         "a figure moved over time rather than one era's best match")
    ap.add_argument("--auto-date", action="store_true",
                    help="infer a date filter from the wording of the query")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    quiet_logs()
    query = " ".join(args.query)
    r = build_retriever(load_config(args.config))

    if args.auto_date and not (args.month or args.date_from or args.date_to):
        latest = max(((m.get("date_original") or "")[:10] for m in r.index.meta),
                     default=None)
        f = infer_filters(query, latest_indexed=latest)
    else:
        f = Filters(date_from=args.date_from, date_to=args.date_to,
                    year_months=args.month, sectors=args.sector,
                    regions=args.region, sources=args.source,
                    require_figures=args.figures_only)

    res = r.search(query, k=args.k, filters=f, max_per_report=args.per_report,
                   max_per_parent=args.per_parent,
                   max_per_period=1 if args.spread else None)

    if args.json:
        print(json.dumps({
            "query": query, "n_candidates": res.n_candidates,
            "n_total": res.n_total, "periods": res.periods,
            "hits": [{"chunk_id": h.chunk_id, "date": h.date_original,
                      "sources": h.sources, "sector": h.sector,
                      "regions": getattr(h, "regions", []), "score": h.score,
                      "cosine": h.dense_score, "url": h.url, "body": h.body}
                     for h in res.hits],
        }, indent=2, ensure_ascii=False))
        return 0

    print(f"\n{query!r}")
    print(f"searched {res.n_candidates} of {res.n_total} chunks", end="")
    print(f"  |  {len(res.hits)} hits" + (f"  |  periods {', '.join(res.periods)}"
                                          if res.periods else ""))
    for note in res.notes:
        print(f"  note: {note}")
    print("=" * 96)
    if not res.hits:
        print("nothing matched. Widen the filters or rephrase.")
        return 1
    for i, h in enumerate(res.hits, start=1):
        print(format_hit(i, h))
        print()
    # Spanning several editions is the normal, correct outcome for an undated
    # question about a serial publication — and the reason every answer must
    # carry an as-of date.
    if res.spans_periods:
        print(f"[{len(res.periods)} reporting periods in these results — any figure "
              "quoted from them needs its date attached]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

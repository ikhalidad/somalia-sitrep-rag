"""
Ask a question. Retrieval, grounded generation, and a citation audit.

    python -m src.ask --list-models
    python -m src.ask "how many cholera cases were reported in Banadir?"
    python -m src.ask "what were displacement figures in March 2025?" --auto-date
    python -m src.ask "what is the funding shortfall?" --sector Funding --show-context

Model names are NOT hardcoded to a guess. Provider lineups change — the model
string that was right six months ago is now enterprise-only on Groq — so the
default is checked against the provider's live model list, and a wrong name
produces the available ones instead of an opaque 404.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap

from .generate import GroundedAnswerer
from .ingest import load_config, quiet_logs
from .retrieve import Filters, infer_filters
from .search import build_retriever

log = logging.getLogger("ask")

PROVIDERS = {
    # base_url, default model, where to get a key
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b",
             "https://console.groq.com"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini",
               "https://platform.openai.com"),
    "together": ("https://api.together.xyz/v1",
                 "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 "https://api.together.xyz"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-oss-120b",
                   "https://openrouter.ai"),
    "local": ("http://localhost:11434/v1", "llama3.1", "ollama / vLLM"),
}


def list_models(base_url: str, api_key: str) -> list[str]:
    import requests

    r = requests.get(f"{base_url}/models",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    return sorted(m.get("id", "") for m in r.json().get("data", []))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.ask")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--config", default="config/corpus.yaml")
    ap.add_argument("--provider", default="groq", choices=sorted(PROVIDERS))
    ap.add_argument("--model", help="override the provider default")
    ap.add_argument("--list-models", action="store_true",
                    help="print the provider's live model list and exit")
    ap.add_argument("--key-env", default="LLM_API_KEY")
    ap.add_argument("-k", type=int, default=6, help="passages to retrieve")
    ap.add_argument("--month", action="append")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--sector", action="append")
    ap.add_argument("--region", action="append")
    ap.add_argument("--source", action="append")
    ap.add_argument("--auto-date", action="store_true",
                    help="infer a date filter from the wording of the question")
    ap.add_argument("--min-score", type=float, default=None,
                    help="abstain threshold on raw cosine (default from generate.py)")
    ap.add_argument("--show-context", action="store_true",
                    help="print the passages the model was given")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    quiet_logs()

    base_url, default_model, key_home = PROVIDERS[args.provider]
    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key:
        print(f"{args.key_env} is not set. Get a key at {key_home}, then:\n"
              f"    export {args.key_env}=...", file=sys.stderr)
        return 2
    model = args.model or default_model

    if args.list_models:
        for m in list_models(base_url, api_key):
            print(("* " if m == model else "  ") + m)
        print(f"\n* = current default for --provider {args.provider}")
        return 0

    if not args.question:
        ap.error("a question is required (or use --list-models)")
    question = " ".join(args.question)

    # Validate before spending a retrieval + a request on an opaque 404.
    try:
        available = list_models(base_url, api_key)
        if available and model not in available:
            print(f"Model {model!r} is not available on {args.provider}.\n"
                  f"Available: {', '.join(available[:12])}"
                  f"{' ...' if len(available) > 12 else ''}\n"
                  f"Pick one with --model, or run --list-models.", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001 — never block on the check itself
        log.warning("could not verify model list (%s); continuing", type(exc).__name__)

    from .generate import MIN_DENSE_SCORE, OpenAICompatLLM

    retriever = build_retriever(load_config(args.config))
    llm = OpenAICompatLLM(model=model, base_url=base_url, api_key_env=args.key_env)
    answerer = GroundedAnswerer(
        retriever, llm, k=args.k,
        min_dense=args.min_score if args.min_score is not None else MIN_DENSE_SCORE,
        auto_filters=args.auto_date,
    )

    filters = None
    if any((args.month, args.date_from, args.date_to, args.sector,
            args.region, args.source)):
        filters = Filters(date_from=args.date_from, date_to=args.date_to,
                          year_months=args.month, sectors=args.sector,
                          regions=args.region, sources=args.source)
    elif args.auto_date:
        filters = infer_filters(question, latest_indexed=answerer._latest)

    ans = answerer.answer(question, filters=filters)

    if args.json:
        print(json.dumps({
            "question": ans.question, "answer": ans.text,
            "abstained": ans.abstained, "flags": ans.flags,
            "trustworthy": ans.trustworthy, "sources": ans.sources,
            "verification": (vars(ans.verification) if ans.verification else None),
        }, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQ: {question}")
    print("=" * 90)
    if args.show_context and ans.retrieval:
        for n, h in enumerate(ans.retrieval.hits, start=1):
            print(f"\n--- context [{n}] {(h.date_original or '')[:10]} "
                  f"{'/'.join(h.sources or [])} ---")
            print(textwrap.fill(" ".join((h.parent_text or h.body).split())[:600],
                                width=88))
        print("\n" + "=" * 90)
    print("\n" + textwrap.fill(ans.text, width=88, replace_whitespace=False) + "\n")

    # Retrieval's caveats qualify the answer, so they belong beside it rather
    # than buried in an object. The era warning in particular is the whole
    # point of the temporal work: an answer drawn entirely from one year looks
    # authoritative and is not.
    if ans.retrieval:
        r_ = ans.retrieval
        print(f"retrieved {len(r_.hits)} of {r_.n_candidates} matching chunks "
              f"({r_.n_total} indexed)"
              + (f"  |  periods {', '.join(r_.periods)}" if r_.periods else ""))
        for note in r_.notes:
            print(f"  note: {note}")
        print()

    if ans.sources:
        print("-" * 90)
        for s in ans.sources:
            mark = "*" if s["cited"] else " "
            print(f"{mark}[{s['n']}] {s['citation']}")
        print("\n* = actually cited in the answer above")

    v = ans.verification
    if v:
        print(f"\ncitation coverage {v.citation_coverage:.0%}"
              f"  |  dates stated: {', '.join(v.dates_mentioned) or 'none'}"
              f"  |  {'REFUSAL' if v.is_refusal else 'answered'}")
    # Verification results are the point, not decoration: an answer that failed
    # the audit must look different from one that passed.
    if ans.flags:
        print(f"\nFLAGS: {', '.join(ans.flags)}")
        print("The answer above did not satisfy the grounding rules it was given. "
              "Treat it as unverified.")
    elif not ans.abstained:
        print("\nverified: every claim cited, every figure dated")
    return 0


if __name__ == "__main__":
    sys.exit(main())

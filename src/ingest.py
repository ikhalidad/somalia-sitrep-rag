"""
Stage 1 + 2 of the pipeline: fetch metadata, resolve document text.

    config/corpus.yaml
            │
            ▼
   [1] fetch   ──► data/raw/reports.jsonl        immutable API payloads
            │      data/raw/pdf/<file_id>.pdf    attachment cache
            ▼
   [2] resolve ──► data/interim/documents.jsonl  one clean text per report
            │      data/interim/manifest.sqlite  resume + provenance
            ▼
        chunker.py (stage 3)

Two rules the rest of the project depends on:

1. Raw is immutable. Everything the API gave us is written to disk before
   any parsing. Re-chunking, re-embedding and re-evaluating never touch the
   network, which matters when the budget is 1000 calls/day and a chunking
   experiment is a thing you want to run twenty times.

2. The manifest is the resume point. Keyed on report_id + date_changed, so
   a re-run fetches only genuinely new or revised documents. A crashed run
   costs you nothing but the calls already spent.

Usage
-----
    export RELIEFWEB_APPNAME='unfpa-som-sitrep-rag-7f3a9c'
    python -m src.ingest sources  --config config/corpus.yaml   # do this first
    python -m src.ingest fetch    --config config/corpus.yaml
    python -m src.ingest resolve  --config config/corpus.yaml
    python -m src.ingest profile  --config config/corpus.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from collections import Counter
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

from .rw_client import ReliefWebClient

log = logging.getLogger("ingest")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    report_id      INTEGER PRIMARY KEY,
    date_original  TEXT,
    date_changed   TEXT,
    title          TEXT,
    sources        TEXT,
    body_chars     INTEGER,
    pdf_file_id    INTEGER,
    pdf_path       TEXT,
    text_source    TEXT,      -- body | pdf | body+pdf | none
    text_chars     INTEGER,
    text_sha256    TEXT,
    extraction_note TEXT,
    fetched_at     TEXT,
    resolved_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_doc_date ON documents(date_original);
"""


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def quiet_logs() -> None:
    """Keep third-party INFO chatter out of the pipeline's own output.

    huggingface_hub and httpx log every HTTP request at INFO, which buries
    the progress lines that make a long run readable.
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "transformers",
                 "urllib3", "filelock", "sentence_transformers", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and normalise the corpus config.

    Date bounds are accepted either at corpus level or inside `filters` and
    normalised into `filters`, because the two read equally naturally in YAML
    and the difference should not be a KeyError three functions deep.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    c = cfg["corpus"]
    f = c.setdefault("filters", {})
    for key in ("date_from", "date_to", "date_field"):
        if key in c and key not in f:
            f[key] = c[key]
    f.setdefault("date_from", "2018-01-01")
    if f.get("date_to") in (None, "null", ""):
        f["date_to"] = date.today().isoformat()
    c.setdefault("date_field", f.get("date_field", "date.original"))
    return cfg


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    p = {k: Path(v) for k, v in cfg["paths"].items()}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def _db(interim: Path) -> sqlite3.Connection:
    con = sqlite3.connect(interim / "manifest.sqlite")
    con.executescript(SCHEMA)
    return con


# --------------------------------------------------------------------------
# stage 1 — fetch
# --------------------------------------------------------------------------

def cmd_fetch(cfg: dict[str, Any], pdf_only: bool = False) -> None:
    """Fetch metadata, then cache attachments.

    `pdf_only` skips the metadata walk. Re-walking nine year-windows to
    discover nothing changed costs ~36 API calls and several minutes, which
    is pure waste when the previous run already completed the metadata and
    only the attachment phase needs resuming.
    """
    paths = _paths(cfg)
    corpus, ex = cfg["corpus"], cfg["extraction"]
    con = _db(paths["interim"])

    known = {
        row[0]: row[1]
        for row in con.execute("SELECT report_id, date_changed FROM documents")
    }
    log.info("manifest holds %d documents", len(known))

    if pdf_only:
        if not known:
            log.error("--pdf-only needs an existing manifest; run fetch first")
            return
        log.info("skipping metadata walk (--pdf-only)")
        if cfg["extraction"]["pdf"]["enabled"]:
            _download_pdfs(cfg, paths, con)
        con.close()
        return

    client = ReliefWebClient(quota_path=paths["raw"].parent / ".rw_quota.json")
    log.info("quota: %d calls remaining today", client.quota.remaining)

    out = paths["raw"] / "reports.jsonl"
    new = revised = skipped = 0
    now = datetime.now(timezone.utc).isoformat()

    with out.open("a", encoding="utf-8") as fh:
        stream = client.iter_windows(
            country_iso3=corpus["filters"]["country_iso3"],
            formats=corpus["filters"]["formats"],
            language_code=corpus["filters"]["language_code"],
            status=corpus["filters"]["status"],
            date_field=corpus["date_field"],
            date_from=date.fromisoformat(corpus["filters"]["date_from"]),
            date_to=date.fromisoformat(corpus["filters"]["date_to"]),
            fields=corpus["fields"],
            interval=corpus.get("slice_interval", "year"),
            sources=corpus["filters"].get("sources"),
            source_field=corpus["filters"].get("source_field", "source.shortname"),
        )
        for rec in stream:
            rid, changed = rec["report_id"], rec.get("date_changed")
            if rid in known:
                if known[rid] == changed:
                    skipped += 1
                    continue
                revised += 1
            else:
                new += 1

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            con.execute(
                """INSERT INTO documents
                   (report_id, date_original, date_changed, title, sources,
                    body_chars, fetched_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(report_id) DO UPDATE SET
                     date_changed=excluded.date_changed,
                     body_chars=excluded.body_chars,
                     fetched_at=excluded.fetched_at,
                     text_source=NULL, resolved_at=NULL""",
                (rid, rec.get("date_original"), changed, rec.get("title"),
                 "; ".join(filter(None, rec.get("sources") or [])),
                 rec.get("body_chars", 0), now),
            )
            if (new + revised) % 200 == 0:
                con.commit()
    con.commit()

    log.info("fetch complete — new=%d revised=%d unchanged=%d | %d calls used",
             new, revised, skipped, client.quota.used)

    if ex["pdf"]["enabled"]:
        _download_pdfs(cfg, paths, con)
    con.close()


def _download_pdfs(cfg, paths: dict[str, Path], con: sqlite3.Connection) -> None:
    """Cache one PDF per report wherever the body cannot stand alone.

    Attachments live on reliefweb.int, not the API host, so these are plain
    file downloads rather than API calls. They are still rate-limited: being
    a good citizen of a free public service costs nothing.
    """
    ex = cfg["extraction"]
    skip_pat = [re.compile(p) for p in ex["pdf"]["skip_filename_patterns"]]
    delay = 1.0 / max(ex["pdf"]["requests_per_second"], 0.1)
    sess = requests.Session()
    # Attachments are served from reliefweb.int, the public website — not from
    # api.reliefweb.int. The appname is an API credential and has no meaning
    # here; sending it as the User-Agent made every request look like an
    # unidentified crawler and the edge returned 404 for all of them. A blanket
    # 404 on every URL, where curl gets 200 on the same URL, is a filter rather
    # than missing files. A conventional UA with a contact handle is both what
    # the host expects and the honest thing to send.
    sess.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) reliefweb-somalia-rag "
                       "(research; contact via GitHub)"),
        "Accept": "application/pdf,*/*",
    })

    primary_only = ex["pdf"].get("primary_country_only", True)
    text_cache = paths["interim"] / "text"

    # -- 1. plan: decide everything before touching the network ------------
    # A silent loop over 2,275 records that downloads 250 of them is
    # indistinguishable from a hang. Planning first means the run can say how
    # much work there is before it starts, and report progress against it.
    plan: list[tuple[int, dict, Path]] = []
    already = skipped_foreign = repaired = 0
    for rec in _read_jsonl(paths["raw"] / "reports.jsonl"):
        if primary_only and not rec.get("is_primary_country_som"):
            skipped_foreign += 1
            continue
        if not _needs_pdf(rec, ex):
            continue
        pdf = _pick_attachment(rec.get("files") or [], skip_pat,
                               ex["pdf"]["accept_mimetypes"])
        if not pdf:
            continue
        dest = paths["pdf_cache"] / f"{pdf['id']}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            if _pdf_looks_complete(dest):
                con.execute("UPDATE documents SET pdf_file_id=?, pdf_path=? "
                            "WHERE report_id=?", (pdf["id"], str(dest), rec["report_id"]))
                already += 1
                continue
            # Truncated by an interrupted earlier run. It looked cached, so it
            # was never re-fetched, and its extraction failed silently. Delete
            # it — and its cached text, or resolve would never re-extract it.
            dest.unlink()
            for stale in (text_cache / f"{pdf['id']}.md",
                          text_cache / f"{pdf['id']}.md.note"):
                stale.unlink(missing_ok=True)
            repaired += 1
        plan.append((rec["report_id"], pdf, dest))
    con.commit()

    log.info("attachments: %d to download, %d already cached, %d non-Somalia "
             "skipped%s", len(plan), already, skipped_foreign,
             f", {repaired} truncated files queued for re-download" if repaired else "")
    if not plan:
        return

    # -- 2. download, with progress -----------------------------------------
    got = failed = consecutive_fail = 0
    t0 = time.monotonic()
    for i, (rid, pdf, dest) in enumerate(plan, start=1):
        # Twenty consecutive failures is never bad luck: it is a blocked UA,
        # a dead network or moved URLs, and retrying politely forever just
        # burns an evening. Stop and say so.
        if consecutive_fail >= 20:
            log.error("20 consecutive attachment failures — aborting. This is "
                      "systemic (blocked UA, network, or moved URLs), not flaky "
                      "downloads. Metadata and completed files are intact.")
            break
        tmp = dest.with_suffix(".part")
        try:
            r = sess.get(pdf["url"], timeout=(10, 45), stream=True)
            r.raise_for_status()
            size = int(r.headers.get("Content-Length") or 0)
            if size > ex["pdf"]["max_mb"] * 1024 * 1024:
                r.close()
                log.info("skip oversized attachment %s (%.1f MB)", pdf["id"], size / 1e6)
            else:
                with tmp.open("wb") as fh:
                    for block in r.iter_content(1 << 16):
                        fh.write(block)
                # Atomic: the .pdf only ever exists complete. A run killed
                # mid-download leaves a .part, which is simply fetched again.
                tmp.replace(dest)
                con.execute("UPDATE documents SET pdf_file_id=?, pdf_path=? "
                            "WHERE report_id=?", (pdf["id"], str(dest), rid))
                got += 1
                consecutive_fail = 0
        except Exception as exc:  # noqa: BLE001 — one bad PDF must not kill a run
            tmp.unlink(missing_ok=True)
            failed += 1
            consecutive_fail += 1
            log.warning("attachment %s failed: %s", pdf.get("id"), exc)

        if i % 25 == 0 or i == len(plan):
            con.commit()
            rate = i / max(time.monotonic() - t0, 1e-6)
            log.info("  downloaded %d/%d  (%d failed)  ~%.0f min left",
                     i, len(plan), failed, (len(plan) - i) / max(rate, 1e-6) / 60)
        time.sleep(delay)

    con.commit()
    log.info("attachments cached=%d failed=%d already=%d skipped_non_somalia=%d",
             got, failed, already, skipped_foreign)


def _pdf_looks_complete(path: Path) -> bool:
    """Cheap truncation check: a PDF opens with %PDF and closes with %%EOF.

    A download interrupted mid-write keeps the header and loses the trailer.
    Checking the last 8 KB tolerates the trailing whitespace some generators
    emit after the final %%EOF.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(5)
            fh.seek(max(size - 8192, 0))
            tail = fh.read()
        return head.startswith(b"%PDF") and b"%%EOF" in tail
    except OSError:
        return False

def _needs_pdf(rec: dict[str, Any], ex: dict[str, Any]) -> bool:
    """Should this report's text come from its PDF rather than its body?

    The first version asked one question: is the body long enough? The
    profile showed that was the wrong question. Body-sourced documents were
    1% structured against 96% for PDFs — and OCHA, which posts long bodies,
    came out at a median of ONE section per document, because ReliefWeb's
    body is a prose rendering with the headings flattened out.

    So length is necessary but not sufficient. A body is used only when it is
    long AND carries its own section structure; otherwise the PDF, which is
    the document as its publisher laid it out, is preferred.
    """
    body = rec.get("body") or ""
    if len(body) < ex["body_min_chars"] or ex.get("merge_body_and_pdf"):
        return True
    if not ex.get("prefer_structured", True):
        return False
    from .chunker import split_sections

    return len(split_sections(body)) < ex.get("min_body_sections", 3)


def _pick_attachment(files, skip_pat, accept_mimes) -> dict | None:
    """Choose the main report PDF, not the annex.

    Heuristic, in order: right mimetype; filename does not match a skip
    pattern; largest remaining by filename length is a poor proxy, so we take
    the FIRST surviving attachment — ReliefWeb editors list the main document
    first with high reliability.
    """
    for f in files:
        if f.get("mimetype") not in accept_mimes or not f.get("url"):
            continue
        name = f"{f.get('filename') or ''} {f.get('description') or ''}"
        if any(p.search(name) for p in skip_pat):
            continue
        return f
    return None


# --------------------------------------------------------------------------
# stage 2 — resolve text
# --------------------------------------------------------------------------

def cmd_resolve(cfg: dict[str, Any]) -> None:
    """Resolve every in-scope report to one clean text.

    Three properties the first version lacked, each learned the hard way:

    RESUMABLE. Extracted markdown is cached per PDF under interim/text/.
    PDF extraction is the slowest step in the pipeline; killing a run at
    document 300 must not mean re-extracting 300 documents.

    PARALLEL and VISIBLE. Extraction runs in a small process pool with a
    progress line every 25 files. A silent single-threaded loop over ~600
    PDFs is indistinguishable from a hang, and gets killed as one.

    SCOPED. The primary-country filter applies here as well as at download.
    Filtering only downloads still let 1,373 non-Somalia teasers into the
    corpus through their body text — the download filter kept their PDFs out
    but not their abstracts.
    """
    paths = _paths(cfg)
    ex = cfg["extraction"]
    con = _db(paths["interim"])
    pdf_paths = dict(
        con.execute("SELECT report_id, pdf_path FROM documents WHERE pdf_path IS NOT NULL")
    )
    primary_only = ex.get("primary_country_only",
                          ex["pdf"].get("primary_country_only", True))
    max_pages = ex["pdf"].get("max_pages", 40)
    engine = ex["pdf"].get("engine", "layout")
    cache_dir = paths["interim"] / "text"
    cache_dir.mkdir(parents=True, exist_ok=True)

    recs = list(_read_jsonl(paths["raw"] / "reports.jsonl"))
    if primary_only:
        before = len(recs)
        recs = [r for r in recs if r.get("is_primary_country_som")]
        log.info("scope: %d Somalia reports kept, %d dropped (tagged Somalia, "
                 "primarily about elsewhere)", len(recs), before - len(recs))

    # -- 1. extract what is not cached yet ---------------------------------
    todo: list[tuple[str, str, int, str]] = []
    for r in recs:
        if not _needs_pdf(r, ex):
            continue
        pp = pdf_paths.get(r["report_id"])
        if not pp or not Path(pp).exists():
            continue
        cp = cache_dir / f"{Path(pp).stem}.md"
        if not cp.exists():
            todo.append((pp, str(cp), max_pages, engine))

    cached_already = sum(1 for _ in cache_dir.glob("*.md"))
    if todo:
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Capped at 3: pymupdf holds whole documents in memory, and WSL gets
        # only a share of system RAM. More workers buys little and risks OOM.
        workers = min(3, max(1, (os.cpu_count() or 2) - 1))
        log.info("extracting %d PDFs with %d workers, engine=%s, OCR off "
                 "(%d already cached)", len(todo), workers, engine, cached_already)
        done = failed = 0
        t0 = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_extract_worker, t) for t in todo]
            for fut in as_completed(futs):
                done += 1
                try:
                    _, n_chars, _ = fut.result()
                    if n_chars == 0:
                        failed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    log.warning("extraction crashed: %s", exc)
                if done % 25 == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - t0, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-6)
                    log.info("  extracted %d/%d  (%d empty)  ~%.0f min left",
                             done, len(todo), failed, eta / 60)
    else:
        log.info("nothing to extract — %d PDFs already cached", cached_already)

    # -- 2. assemble documents from cache (fast) ---------------------------
    out = paths["interim"] / "documents.jsonl"
    counts: Counter = Counter()
    now = datetime.now(timezone.utc).isoformat()
    with out.open("w", encoding="utf-8") as fh:
        for rec in recs:
            text, origin, note = _resolve_one(
                rec, pdf_paths.get(rec["report_id"]), ex, cache_dir)
            counts[origin] += 1
            if not text.strip():
                continue
            doc = {**{k: v for k, v in rec.items() if k != "body"},
                   "text": text, "text_source": origin, "extraction_note": note}
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            con.execute(
                """UPDATE documents SET text_source=?, text_chars=?, text_sha256=?,
                   extraction_note=?, resolved_at=? WHERE report_id=?""",
                (origin, len(text), hashlib.sha256(text.encode()).hexdigest(),
                 note, now, rec["report_id"]),
            )
    con.commit()
    con.close()
    log.info("resolved %d documents: %s", sum(counts.values()), dict(counts))


def _extract_worker(task: tuple[str, str, int, str]) -> tuple[str, int, str]:
    """Extract one PDF into the cache. Top-level so it pickles.

    Python 3.14 defaults to the forkserver start method on Linux, which
    imports the worker function by name in a fresh interpreter — a nested
    function or lambda here fails with an opaque pickling error.

    Failures are cached as an empty file with a `.note` beside it, so a
    scanned PDF is attempted once rather than on every re-run. Delete the
    pair to force a retry.
    """
    pdf_path, cache_path, max_pages, engine = task
    text, note = extract_pdf_markdown(Path(pdf_path), max_pages=max_pages,
                                      engine=engine)
    Path(cache_path).write_text(text, encoding="utf-8")
    if note:
        Path(cache_path + ".note").write_text(note, encoding="utf-8")
    return pdf_path, len(text), note


def _resolve_one(rec, pdf_path, ex, cache_dir: Path) -> tuple[str, str, str]:
    body = (rec.get("body") or "").strip()
    if not _needs_pdf(rec, ex):
        return body, "body", ""

    pdf_text, note = "", "no attachment cached"
    if pdf_path:
        cp = cache_dir / f"{Path(pdf_path).stem}.md"
        if cp.exists():
            pdf_text = cp.read_text(encoding="utf-8")
            np_ = Path(str(cp) + ".note")
            note = np_.read_text(encoding="utf-8") if np_.exists() else ""

    if ex["merge_body_and_pdf"] and body and pdf_text:
        return f"{body}\n\n{pdf_text}", "body+pdf", note
    if pdf_text:
        return pdf_text, "pdf", note
    if body:
        return body, "body", f"short body ({len(body)} chars); {note}".strip("; ")
    return "", "none", note


def extract_pdf_markdown(path: Path, max_pages: int = 40,
                         engine: str = "layout") -> tuple[str, str]:
    """PDF -> Markdown, preserving headings and tables.

    OCR is always OFF. Situation reports are born-digital: every word is
    already in the text layer. OCR adds nothing but cost and noise — it reads
    the labels off maps and chart axes, so a map of Somalia injects "Gedo Bay
    Bakool Banadir" into the extracted text and the region classifier then
    tags a section with regions it never discusses. It is also the difference
    between ~0.3s and several seconds per page.

    Two engines. `layout` (default) runs pymupdf's ML layout model — slower,
    and the one that produced median 9 sections/doc on real WHO bulletins in
    the first profile run. `legacy` uses font-size heading detection — ~5x
    faster, same heading yield on the test fixtures, untested on real PDFs.
    Switch in config and compare the `profile` sections/doc figure.

    Work per file is bounded by page count, not a wall-clock timeout: sitreps
    run 2-15 pages; the files that stall extraction are 200-page annex
    compilations, and capping pages removes them without killing processes.

    Licensing: PyMuPDF is AGPL-3.0. Fine for an open-source portfolio
    project; swap to pdfplumber (MIT) if permissive licensing is needed.
    """
    import contextlib
    import io

    try:
        try:
            import pymupdf  # type: ignore
        except ImportError:
            import fitz as pymupdf  # type: ignore
        import pymupdf4llm  # type: ignore

        if engine == "legacy" and hasattr(pymupdf4llm, "use_layout"):
            pymupdf4llm.use_layout(False)

        with pymupdf.open(str(path)) as doc:
            n_pages = doc.page_count
        pages = list(range(min(n_pages, max_pages)))

        # The layout engine prints parser banners to stdout for every file;
        # across 600 files that buries the progress lines that make a long
        # run legible.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                md = pymupdf4llm.to_markdown(str(path), pages=pages,
                                             show_progress=False, use_ocr=False)
            except TypeError:
                # Pre-1.x pymupdf4llm has no use_ocr — and no OCR either.
                md = pymupdf4llm.to_markdown(str(path), pages=pages,
                                             show_progress=False)

        note = (f"truncated to first {max_pages} of {n_pages} pages"
                if n_pages > max_pages else "")
        if len(md.strip()) < 200:
            return "", "extraction yielded <200 chars — likely a scanned PDF"
        return md, note
    except ImportError:
        return "", "pymupdf4llm not installed"
    except Exception as exc:  # noqa: BLE001
        return "", f"pdf extraction failed: {type(exc).__name__}"


# --------------------------------------------------------------------------
# profile — look before you index
# --------------------------------------------------------------------------

def cmd_sources(cfg: dict[str, Any]) -> None:
    """Publisher counts for the corpus, ignoring the source filter.

    Two calls total. Run this BEFORE the first fetch. It answers the two
    questions the config can only guess at: what ReliefWeb actually calls
    each publisher (source.shortname is exact-matched, so 'WHO' vs 'World
    Health Organization' is the difference between a corpus and an empty
    file), and what you give up by narrowing to three of them.
    """
    paths = _paths(cfg)
    f = cfg["corpus"]["filters"]
    client = ReliefWebClient(quota_path=paths["raw"].parent / ".rw_quota.json")

    unfiltered = client.build_filter(
        f["country_iso3"], f["formats"], f["language_code"], f["status"],
        cfg["corpus"]["date_field"],
        date.fromisoformat(f["date_from"]), date.fromisoformat(f["date_to"]),
    )
    rows = client.facet(unfiltered, f.get("source_field", "source.shortname"), limit=60)
    total = sum(n for _, n in rows)
    chosen = set(f.get("sources") or [])

    print(f"Somalia situation reports, {f['date_from']} to {f['date_to']}")
    print(f"{total} documents across {len(rows)} publishers\n")
    print(f"{'':2} {'publisher':<14} {'docs':>6}  {'share':>6}")
    kept = 0
    for name, n in rows:
        mark = "->" if name in chosen else "  "
        kept += n if name in chosen else 0
        print(f"{mark} {str(name):<14} {n:>6}  {n / max(total,1):>5.0%}")

    print(f"\nselected {len(chosen & {r[0] for r in rows})}/{len(chosen)} configured "
          f"publishers -> {kept} docs ({kept / max(total,1):.0%} of the corpus)")
    missing = chosen - {r[0] for r in rows}
    if missing:
        print(f"\nWARNING: no match for {sorted(missing)} — these contribute ZERO "
              f"documents. Check the spelling against the list above before fetching.")
    if kept < 150:
        print("\nNOTE: under ~150 documents is thin for a retrieval demo. Consider "
              "widening the date range or adding a publisher.")


def cmd_profile(cfg: dict[str, Any]) -> None:
    """Corpus shape report. Run this BEFORE tuning body_min_chars."""
    paths = _paths(cfg)
    recs = list(_read_jsonl(paths["raw"] / "reports.jsonl"))
    if not recs:
        print("nothing fetched yet")
        return

    lens = sorted(r["body_chars"] for r in recs)
    by_source: dict[str, int] = {}
    by_year: dict[str, int] = {}
    with_pdf = 0
    for r in recs:
        for s in r.get("source_shortnames") or r.get("sources") or []:
            if s:
                by_source[s] = by_source.get(s, 0) + 1
        y = (r.get("date_original") or "????")[:4]
        by_year[y] = by_year.get(y, 0) + 1
        if any(f.get("mimetype") == "application/pdf" for f in r.get("files") or []):
            with_pdf += 1

    def pct(p: float) -> int:
        return lens[min(int(len(lens) * p), len(lens) - 1)]

    som = sum(1 for r in recs if r.get("is_primary_country_som"))
    foreign: Counter = Counter()
    for r in recs:
        if not r.get("is_primary_country_som"):
            foreign[r.get("primary_country") or "?"] += 1

    print(f"documents            {len(recs)}")
    print(f"primary country SOM  {som} ({som / len(recs):.0%})"
          f"   <- the actual Somalia corpus")
    if foreign:
        top = ", ".join(f"{k} {v}" for k, v in foreign.most_common(6))
        print(f"tagged SOM but about {sum(foreign.values())} others: {top}")
    print(f"with PDF attachment  {with_pdf} ({with_pdf / len(recs):.0%})")
    print(f"body chars  p10={pct(.10)}  p50={pct(.50)}  p90={pct(.90)}  max={lens[-1]}")
    print(f"bodies under 2500ch  {sum(l < 2500 for l in lens)} "
          f"({sum(l < 2500 for l in lens) / len(lens):.0%})  <- these need the PDF")
    print("\nby year:")
    for y in sorted(by_year):
        print(f"  {y}  {by_year[y]:>5}")
    print("\ntop sources:")
    for s, n in sorted(by_source.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:>5}  {s}")

    # Structure is measured on the best text available. Before `resolve`
    # that is the raw `body`, which for this corpus is a teaser with no
    # headings — measuring it tells you about ReliefWeb's summary field, not
    # about the reports. After `resolve` it is the extracted PDF markdown,
    # which is where the section skeleton actually lives.
    resolved = paths["interim"] / "documents.jsonl"
    if resolved.exists():
        docs = list(_read_jsonl(resolved))
        by_src = Counter(d.get("text_source") for d in docs)
        lens = sorted(len(d.get("text") or "") for d in docs)
        in_scope = sum(1 for r in recs if r.get("is_primary_country_som"))
        need_pdf = sum(1 for r in recs if r.get("is_primary_country_som")
                       and _needs_pdf(r, cfg["extraction"]))
        pdf_dir = paths["pdf_cache"]
        n_pdf = sum(1 for _ in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else 0
        txt_dir = paths["interim"] / "text"
        n_txt = sum(1 for f in txt_dir.glob("*.md") if f.stat().st_size) if txt_dir.exists() else 0
        print(f"\n[coverage]")
        print(f"  Somalia reports in scope    {in_scope}")
        print(f"  ...needing PDF text         {need_pdf}   <- thin OR unstructured body")
        print(f"  PDFs downloaded             {n_pdf}")
        print(f"  PDFs extracted to text      {n_txt}")
        print(f"  documents resolved          {len(docs)}")
        if len(docs) < in_scope * 0.95 or n_txt < need_pdf * 0.9:
            print("  INCOMPLETE: resolve or the PDF download has not finished. Run "
                  "`fetch --pdf-only` then `resolve` to completion before reading "
                  "the structure figures below — they describe a partial corpus.")
        print(f"\n[resolved] {len(docs)} documents carry extracted text")
        print(f"  text source   {dict(by_src)}")
        print(f"  text chars    p10={lens[len(lens)//10]}  p50={lens[len(lens)//2]}"
              f"  p90={lens[min(len(lens)*9//10, len(lens)-1)]}  max={lens[-1]}")
        _profile_structure(docs, text_key="text")
    else:
        print("\n[structure] measuring the RAW body field — `resolve` has not run, "
              "so no PDF text exists yet")
        _profile_structure(recs, text_key="body")


def _profile_structure(recs: list[dict[str, Any]], text_key: str = "body",
                       min_chars: int = 800) -> None:
    """Does the section vocabulary actually match this corpus?

    The chunker's whole advantage rests on splitting documents at real
    section boundaries. That assumption was formed against OCHA's template;
    this measures it against whatever publishers actually dominate. If most
    documents yield one section, or most headings match no known sector, the
    chunker has quietly degraded to fixed-size splitting and the fix is to
    extend SECTOR_PATTERNS — with the headings printed below.
    """
    from .chunker import classify_regions, is_furniture, split_sections

    usable = [r for r in recs if len(r.get(text_key) or "") >= min_chars]
    if not usable:
        print(f"\n[structure] no documents with >={min_chars} chars of "
              f"{'extracted text' if text_key == 'text' else 'body'} to analyse")
        return

    sec_counts: list[int] = []
    matched = unmatched = furniture = regional = 0
    unmatched_headings: Counter = Counter()
    by_pub: dict[str, list[int]] = {}
    by_origin: dict[str, list[int]] = {}

    for r in usable:
        secs = split_sections(r[text_key])
        sec_counts.append(len(secs))
        pub = next(iter(r.get("source_shortnames") or ["?"]), "?") or "?"
        by_pub.setdefault(pub, []).append(len(secs))
        by_origin.setdefault(r.get("text_source") or text_key, []).append(len(secs))
        for s in secs:
            h = (s.heading or "").strip()
            if s.sector:
                matched += 1
            elif s.furniture:
                furniture += 1          # dropped at chunk time; not a gap
            elif h and classify_regions(h) and len(h.split()) <= 3:
                regional += 1           # captured by the region facet
            else:
                unmatched += 1
                if h and h != "Preamble" and len(h) < 60:
                    unmatched_headings[h] += 1

    sec_counts.sort()
    total_secs = matched + unmatched + furniture + regional
    multi = sum(1 for c in sec_counts if c >= 3)

    print(f"\n[structure] {len(usable)} documents with >={min_chars} chars of body")
    print(f"  sections/doc   p50={sec_counts[len(sec_counts)//2]}  "
          f"p90={sec_counts[min(len(sec_counts)*9//10, len(sec_counts)-1)]}  "
          f"max={sec_counts[-1]}")
    print(f"  docs with >=3 sections   {multi} ({multi/len(usable):.0%})"
          f"   <- structure-aware chunking only helps these")
    print(f"  sections accounted for   {matched + furniture + regional}/{total_secs} "
          f"({(matched + furniture + regional)/max(total_secs,1):.0%})"
          f"   sector {matched} + region {regional} + furniture {furniture}")

    # The decisive split. Structure lives in PDFs, not in ReliefWeb's body
    # field; if PDF-sourced documents are well structured and body-sourced
    # ones are not, the remedy is more PDF coverage, not more vocabulary.
    print("\n  by text source (share with >=3 sections, median sections/doc):")
    for origin, counts in sorted(by_origin.items(), key=lambda x: -len(x[1])):
        counts.sort()
        share = sum(1 for c in counts if c >= 3) / len(counts)
        print(f"    {origin:<10} n={len(counts):<5} {share:>4.0%} structured   "
              f"median={counts[len(counts)//2]}")

    print("\n  by publisher (median sections/doc):")
    for pub, counts in sorted(by_pub.items(), key=lambda x: -len(x[1]))[:8]:
        counts.sort()
        print(f"    {pub:<14} n={len(counts):<5} median={counts[len(counts)//2]}")

    if unmatched_headings:
        print("\n  most common headings matching NO sector "
              "— candidates for SECTOR_PATTERNS:")
        for h, n in unmatched_headings.most_common(30):
            print(f"    {n:>5}  {h}")

    pdf_counts = by_origin.get("pdf", [])
    pdf_share = (sum(1 for c in pdf_counts if c >= 3) / len(pdf_counts)) if pdf_counts else 0
    if multi / len(usable) < 0.5:
        if pdf_counts and pdf_share >= 0.6:
            print(f"\n  NOTE: {pdf_share:.0%} of PDF-sourced documents are structured; "
                  "the low overall share comes from body-sourced teasers. The chunker "
                  "is working — the gap is PDF coverage, not heading vocabulary.")
        else:
            print("\n  WARNING: most documents yield fewer than 3 sections, including "
                  "PDF-sourced ones. The heading patterns do not fit these publishers "
                  "yet — extend SECTOR_PATTERNS from the list above.")


# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Read JSONL, last-write-wins per report_id (append-only raw file)."""
    if not path.exists():
        return []
    latest: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                latest[rec["report_id"]] = rec
    return latest.values()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.ingest")
    ap.add_argument("command",
                    choices=["sources", "fetch", "resolve", "profile"])
    ap.add_argument("--config", default="config/corpus.yaml")
    ap.add_argument("--pdf-only", action="store_true",
                    help="fetch: resume attachment downloads, skip the metadata walk")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    quiet_logs()
    cfg = load_config(args.config)
    if args.command == "fetch":
        cmd_fetch(cfg, pdf_only=args.pdf_only)
    else:
        {"sources": cmd_sources, "resolve": cmd_resolve,
         "profile": cmd_profile}[args.command](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

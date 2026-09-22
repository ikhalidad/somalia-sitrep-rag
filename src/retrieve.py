"""
Stage 5: retrieval.

This is where the chunking decisions either pay off or do not, so the design
is stated plainly.

Filter BEFORE scoring, not after
--------------------------------
The obvious implementation retrieves top-k and then discards anything outside
the requested date range. That is wrong in a way that is easy to miss: if the
question is about March 2025 and the top 50 results are all 2019 editions of
the same paragraph, post-filtering returns an empty list from a corpus that
holds the answer. Pre-filtering builds a boolean mask first and takes top-k
*within* the surviving rows, so k results means k relevant results.

This is the operational half of the temporal-collision fix. The chunker put
the date in the text and in the metadata; this module is what uses it.

Hybrid dense + BM25, fused by RRF
---------------------------------
Reciprocal Rank Fusion rather than score interpolation, because cosine
similarity and BM25 scores live on unrelated scales and normalising them
against each other requires a calibration constant nobody can defend. RRF
fuses on rank alone, needs one parameter, and is stable across query types.

The split matters on this corpus. "What was the cholera case fatality rate in
Banadir?" is a lexical query wearing a semantic costume: BM25 nails
`banadir`, `cholera`, `fatality`; dense retrieval finds the health-cluster
register but drifts across regions. "How is the humanitarian situation
deteriorating?" is the reverse. Fusion covers both without a query classifier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embed import ChunkIndex, Embedder, tokenize

log = logging.getLogger("retrieve")

RRF_K = 60  # standard constant; dampens the influence of any single ranker


@dataclass
class Filters:
    """Metadata predicates applied before scoring."""

    date_from: str | None = None       # 'YYYY-MM-DD' or 'YYYY-MM'
    date_to: str | None = None
    year_months: Sequence[str] | None = None   # exact months, e.g. ['2025-03']
    sectors: Sequence[str] | None = None
    regions: Sequence[str] | None = None
    sources: Sequence[str] | None = None
    require_figures: bool = False
    primary_country_only: bool = False

    def is_empty(self) -> bool:
        return not any((self.date_from, self.date_to, self.year_months,
                        self.sectors, self.regions, self.sources,
                        self.require_figures, self.primary_country_only))


@dataclass
class Hit:
    chunk_id: str
    score: float
    body: str
    header: str
    date_original: str | None
    sources: list[str]
    sector: str | None
    title: str | None
    url: str | None
    report_id: int
    parent_id: str
    rank_dense: int | None = None
    rank_lexical: int | None = None
    dense_score: float = 0.0   # raw cosine — the only calibrated signal here
    parent_text: str | None = None

    def citation(self) -> str:
        d = (self.date_original or "")[:10]
        src = "/".join(self.sources or []) or "ReliefWeb"
        return f"{src}, {d} — {self.title or 'Situation Report'} ({self.url})"


@dataclass
class RetrievalResult:
    hits: list[Hit]
    n_candidates: int          # rows surviving the filter
    n_total: int               # rows in the index
    filters_applied: Filters
    notes: list[str] = field(default_factory=list)
    suppressed: int = 0        # near-duplicates and over-quota hits dropped

    @property
    def periods(self) -> list[str]:
        """Distinct year-months in the result set, oldest first."""
        return sorted({h.date_original[:7] for h in self.hits if h.date_original})

    @property
    def spans_periods(self) -> bool:
        """True when the context mixes editions — the generator must date figures."""
        return len(self.periods) > 1

    @property
    def max_dense(self) -> float:
        return max((h.dense_score for h in self.hits), default=0.0)


class Retriever:
    def __init__(
        self,
        index: ChunkIndex,
        embedder: Embedder,
        parents: dict[str, dict] | None = None,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.parents = parents or {}
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self._dates = np.array(
            [(m.get("date_original") or "")[:10] for m in index.meta], dtype=object
        )
        self.periods_indexed = sorted({m.get("year_month") for m in index.meta
                                       if m.get("year_month")})

    # -- filtering ----------------------------------------------------------

    def _mask(self, f: Filters) -> np.ndarray:
        n = len(self.index)
        mask = np.ones(n, dtype=bool)
        if f.is_empty():
            return mask

        meta = self.index.meta
        if f.date_from:
            lo = _pad_date(f.date_from, end=False)
            mask &= np.array([bool(d) and d >= lo for d in self._dates])
        if f.date_to:
            hi = _pad_date(f.date_to, end=True)
            mask &= np.array([bool(d) and d <= hi for d in self._dates])
        if f.year_months:
            want = set(f.year_months)
            mask &= np.array([m.get("year_month") in want for m in meta])
        if f.sectors:
            want = {s.lower() for s in f.sectors}
            mask &= np.array([(m.get("sector") or "").lower() in want for m in meta])
        if f.regions:
            want = {r.lower() for r in f.regions}
            mask &= np.array([
                bool(want & {str(x).lower() for x in (m.get("regions") or [])})
                for m in meta
            ])
        if f.sources:
            want = {s.lower() for s in f.sources}
            mask &= np.array([
                bool(want & {str(x).lower() for x in (m.get("sources") or [])})
                for m in meta
            ])
        if f.require_figures:
            mask &= np.array([bool(m.get("contains_figures")) for m in meta])
        if f.primary_country_only:
            mask &= np.array([bool(m.get("is_primary_country_som")) for m in meta])
        return mask

    # -- scoring ------------------------------------------------------------

    def _dense_ranks(
        self, query: str, cand: np.ndarray, k: int
    ) -> tuple[list[int], dict[int, float]]:
        qv = self.embedder.encode([query], is_query=True)[0]
        sims = self.index.vectors[cand] @ qv          # unit vectors => cosine
        top = np.argsort(-sims)[:k]
        ranked = [int(cand[i]) for i in top]
        # RRF output is ordinal: it reports the best of whatever survived the
        # filter and cannot distinguish a strong match from the least bad of a
        # bad set. The abstain gate in generate.py needs an absolute signal, so
        # the raw cosine is carried through alongside the fused rank.
        return ranked, {int(cand[i]): float(sims[i]) for i in top}

    def _lexical_ranks(self, query: str, cand: np.ndarray, k: int) -> list[int]:
        if self.index.bm25 is None:
            return []
        scores = np.asarray(self.index.bm25.get_scores(tokenize(query)))
        sub = scores[cand]
        top = np.argsort(-sub)[:k]
        # BM25 returns 0.0 for rows sharing no term with the query. Keeping
        # them would pad the fusion with arbitrary rows that outrank genuine
        # dense matches purely by occupying a lexical rank slot.
        return [int(cand[i]) for i in top if sub[i] > 0]

    def search(
        self,
        query: str,
        k: int = 6,
        filters: Filters | None = None,
        candidate_k: int = 50,
        expand_parents: bool = True,
        max_per_report: int = 2,
        max_per_parent: int = 1,
        max_per_period: int | None = None,
    ) -> RetrievalResult:
        f = filters or Filters()
        notes: list[str] = []
        mask = self._mask(f)
        cand = np.flatnonzero(mask)

        if cand.size == 0:
            # Fail loudly rather than silently widening. A retriever that
            # quietly ignores a date filter will answer a March question with
            # a 2019 figure, which is the precise failure this design exists
            # to prevent.
            return RetrievalResult([], 0, len(self.index), f,
                                   ["No chunks match the filter. Widen the date "
                                    "range or drop a constraint."])
        if cand.size < k:
            notes.append(f"only {cand.size} chunks survive the filter")

        dense, dense_scores = self._dense_ranks(query, cand, candidate_k)
        lexical = self._lexical_ranks(query, cand, candidate_k)
        if not lexical and self.index.bm25 is not None:
            notes.append("no lexical matches — dense-only for this query")

        fused = self._rrf(dense, lexical)
        picked, suppressed = self._select(fused, k, max_per_report, max_per_parent,
                                          max_per_period)
        hits = [self._hit(i, s, dense, lexical, dense_scores) for i, s in picked]
        if suppressed:
            notes.append(f"{suppressed} near-duplicate or over-quota results suppressed")

        # An undated question about a serial corpus can return a single era
        # and look authoritative. The 2019 cholera outbreak dominates any
        # "cholera in Banadir" query on similarity alone, and a reader has no
        # way to tell that six further years exist. Say so.
        if f.is_empty() and self.periods_indexed:
            span = self._span(hits)
            if span and span[0][:4] == span[1][:4] and self.periods_indexed[0][:4] != \
                    self.periods_indexed[-1][:4]:
                notes.append(
                    f"all results fall in {span[0]}..{span[1]}, but the corpus spans "
                    f"{self.periods_indexed[0]}..{self.periods_indexed[-1]} — add a "
                    "date filter to ask about another period")

        if expand_parents and self.parents:
            for h in hits:
                p = self.parents.get(h.parent_id)
                if p:
                    h.parent_text = p.get("text")

        return RetrievalResult(hits, int(cand.size), len(self.index), f, notes)

    def _select(
        self, fused: list[tuple[int, float]], k: int,
        max_per_report: int, max_per_parent: int, max_per_period: int | None,
    ) -> tuple[list[tuple[int, float]], int]:
        """Pick k results under diversity constraints, then backfill.

        Three ways one answer crowds out the rest of the evidence:

        - the SAME text republished in a later edition (tagged at chunk time
          with a shared content_group, so the index keeps every date
          addressable while the result set shows it once);
        - CONSECUTIVE chunks of one section, which overlap by construction.
          A prefix check misses these: the shared text sits at the end of one
          and the start of the next. Overlap only ever occurs inside a
          section, so parent_id is the exact guard;
        - one report taking every slot.

        Constraints are preferences, not hard limits: if they cannot fill k,
        the remainder is backfilled in rank order rather than returning fewer
        results than the caller asked for.
        """
        meta = self.index.meta
        picked: list[tuple[int, float]] = []
        rejected: list[tuple[int, float]] = []
        per_report: dict[Any, int] = {}
        per_parent: dict[Any, int] = {}
        per_period: dict[Any, int] = {}
        seen_groups: set[str] = set()
        seen_openings: set[str] = set()

        for idx, score in fused:
            if len(picked) >= k:
                break
            m = meta[idx]
            group = m.get("content_group")
            # First 160 normalised characters. Overlapping chunks from one
            # document share their opening by construction, and two results
            # whose first paragraph is identical are one result.
            opening = re.sub(r"\s+", " ", (m.get("body") or "")[:160]).strip().lower()
            period = m.get("year_month")
            report = m.get("report_id")
            parent = m.get("parent_id")

            if ((group and group in seen_groups)
                    or (opening and opening in seen_openings)
                    or per_parent.get(parent, 0) >= max_per_parent
                    or per_report.get(report, 0) >= max_per_report
                    or (max_per_period and per_period.get(period, 0) >= max_per_period)):
                rejected.append((idx, score))
                continue

            picked.append((idx, score))
            if group:
                seen_groups.add(group)
            if opening:
                seen_openings.add(opening)
            per_report[report] = per_report.get(report, 0) + 1
            per_parent[parent] = per_parent.get(parent, 0) + 1
            per_period[period] = per_period.get(period, 0) + 1

        suppressed = len(rejected)
        for item in rejected:                      # backfill in rank order
            if len(picked) >= k:
                break
            picked.append(item)
            suppressed -= 1
        return picked, suppressed

    @staticmethod
    def _span(hits: list[Hit]) -> tuple[str, str] | None:
        months = sorted(h.date_original[:7] for h in hits if h.date_original)
        return (months[0], months[-1]) if months else None

    def _rrf(self, dense: list[int], lexical: list[int]) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for rank, idx in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + self.dense_weight / (RRF_K + rank + 1)
        for rank, idx in enumerate(lexical):
            scores[idx] = scores.get(idx, 0.0) + self.lexical_weight / (RRF_K + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def _hit(self, i: int, score: float, dense, lexical, dense_scores) -> Hit:
        m = self.index.meta[i]
        return Hit(
            chunk_id=m["chunk_id"], score=round(float(score), 6),
            body=m.get("body", ""), header=m.get("header", ""),
            date_original=m.get("date_original"), sources=m.get("sources") or [],
            sector=m.get("sector"), title=m.get("title"), url=m.get("url"),
            report_id=m.get("report_id"), parent_id=m.get("parent_id"),
            rank_dense=dense.index(i) + 1 if i in dense else None,
            rank_lexical=lexical.index(i) + 1 if i in lexical else None,
            dense_score=round(dense_scores.get(i, 0.0), 4),
        )


# --------------------------------------------------------------------------
# lightweight temporal parsing
# --------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_YEAR_RE = re.compile(r"\b(20[0-3]\d)\b")
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.I)
_RECENT_RE = re.compile(r"\b(latest|most recent|current|now|today|as of now)\b", re.I)


def infer_filters(query: str, latest_indexed: str | None = None) -> Filters:
    """Pull an explicit time reference out of the question.

    Deliberately conservative: it only fires on a date the user actually
    wrote, or on an explicit recency word. Guessing a time window the user
    did not ask for trades one silent failure for another — and an undated
    question about a serial publication is better answered by returning
    several editions and letting the generator state the as-of date than by
    the retriever inventing a window.
    """
    f = Filters()
    year = _YEAR_RE.search(query)
    month = _MONTH_RE.search(query)

    if year and month:
        y, m = int(year.group(1)), _MONTHS[month.group(1).lower()]
        f.year_months = [f"{y:04d}-{m:02d}"]
    elif year:
        y = int(year.group(1))
        f.date_from, f.date_to = f"{y}-01-01", f"{y}-12-31"
    elif _RECENT_RE.search(query) and latest_indexed:
        # "latest" means the most recent edition, not the last 90 days of
        # whatever happens to be indexed.
        anchor = date.fromisoformat(latest_indexed[:10])
        f.date_from = f"{anchor.year:04d}-{max(anchor.month - 2, 1):02d}-01"
    return f


def _pad_date(s: str, end: bool) -> str:
    """'2025' -> '2025-01-01' / '2025-12-31'; '2025-03' -> month bounds."""
    parts = s.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-12-31" if end else f"{parts[0]}-01-01"
    if len(parts) == 2:
        if not end:
            return f"{s}-01"
        y, m = int(parts[0]), int(parts[1])
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        return (nxt - __import__("datetime").timedelta(days=1)).isoformat()
    return s


def load_parents(path: str | Path) -> dict[str, dict]:
    import json

    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["parent_id"]] = rec
    return out

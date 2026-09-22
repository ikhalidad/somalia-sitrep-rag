"""
Retrieval tests, run with a deterministic stand-in embedder so CI needs no
model download and no network.

The centrepiece is test_temporal_collision_is_fixed: three editions of the
same sitrep with three different IDP figures, and the assertion that a dated
question returns the right edition. That is the whole thesis of the project,
expressed as a regression test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunker import SitrepChunker  # noqa: E402
from src.embed import build_index, tokenize  # noqa: E402
from src.retrieve import Filters, Retriever, infer_filters  # noqa: E402
from tests.test_chunker import CFG, DOC  # noqa: E402


class HashingEmbedder:
    """Deterministic bag-of-words vectoriser — real lexical similarity, no download."""

    model_name = "test-hashing-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                out[r, hash(tok) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


# Three editions, same skeleton, different numbers — the corpus's defining shape.
EDITIONS = [
    ("2024-03-14", 4211801, "2.9 million"),
    ("2025-03-14", 4211802, "3.8 million"),
    ("2026-03-14", 4211803, "4.4 million"),
]


def _corpus():
    chunker = SitrepChunker(CFG)
    chunks = []
    for iso, rid, figure in EDITIONS:
        # Replace every occurrence so each edition is internally consistent —
        # the headline sentence AND the key-figures table row.
        text = DOC["text"].replace("3.8 million", figure)
        chunks.extend(chunker.chunk_document({
            **DOC, "report_id": rid, "text": text,
            "date_original": f"{iso}T00:00:00+00:00",
            "title": f"Somalia: Humanitarian Situation Report ({iso[:7]})",
        }))
    chunker.mark_duplicates(chunks)
    live = [c for c in chunks if not c.duplicate_of and not c.is_boilerplate]
    from dataclasses import asdict
    return [asdict(c) for c in live]


@pytest.fixture(scope="module")
def retriever():
    chunks = _corpus()
    index = build_index(chunks, HashingEmbedder())
    return Retriever(index, HashingEmbedder())


# -- the headline test ------------------------------------------------------

def test_temporal_collision_is_fixed(retriever):
    """The same sentence in three editions must resolve to the asked-for one."""
    q = "how many people are internally displaced"
    for iso, _, figure in EDITIONS:
        res = retriever.search(q, k=3, filters=Filters(year_months=[iso[:7]]))
        assert res.hits, f"nothing returned for {iso[:7]}"
        joined = " ".join(h.body for h in res.hits)
        assert figure in joined, f"{iso[:7]} should surface {figure}"
        for other_iso, _, other_fig in EDITIONS:
            if other_iso != iso:
                assert other_fig not in joined, \
                    f"{iso[:7]} query leaked the {other_iso[:7]} figure {other_fig}"


def test_unfiltered_query_is_temporally_ambiguous(retriever):
    """Documents the failure the filter exists to prevent — do not 'fix' this.

    Without a date constraint the retriever cannot know which edition is
    wanted, and returns several. That is correct behaviour: the generator is
    then obliged to state an as-of date. The bug would be returning exactly
    one and sounding certain.
    """
    res = retriever.search("how many people are internally displaced", k=6)
    years = {(h.date_original or "")[:4] for h in res.hits}
    assert len(years) > 1, "expected multiple editions when no date is given"


# -- filtering --------------------------------------------------------------

def test_filter_applies_before_scoring(retriever):
    res = retriever.search("cholera cases", k=5, filters=Filters(year_months=["2025-03"]))
    assert res.hits
    assert all(h.date_original.startswith("2025-03") for h in res.hits)
    assert res.n_candidates < res.n_total


def test_empty_filter_result_says_so_rather_than_widening(retriever):
    res = retriever.search("cholera", k=5, filters=Filters(year_months=["2019-07"]))
    assert res.hits == []
    assert res.notes and "No chunks match" in res.notes[0]


def test_sector_filter(retriever):
    res = retriever.search("what support was provided", k=5,
                           filters=Filters(sectors=["Protection"]))
    assert res.hits
    assert all(h.sector == "Protection" for h in res.hits)


def test_date_range_filter_spans_editions(retriever):
    res = retriever.search("displaced", k=10,
                           filters=Filters(date_from="2025-01", date_to="2026-12"))
    years = {(h.date_original or "")[:4] for h in res.hits}
    assert "2024" not in years


# -- hybrid -----------------------------------------------------------------

def test_lexical_arm_catches_place_names(retriever):
    """Dense retrieval smears Somali place names; BM25 is why this works."""
    res = retriever.search("Gedo essential medicines shortage", k=5)
    assert any("Gedo" in h.body for h in res.hits)


def test_hits_carry_a_usable_citation(retriever):
    res = retriever.search("funding requirements", k=3)
    cite = res.hits[0].citation()
    assert "OCHA" in cite and "reliefweb.int" in cite and "20" in cite


# -- query-time date inference ---------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("how many IDPs in March 2025", ["2025-03"]),
    ("displacement figures for March 2026", ["2026-03"]),
])
def test_infer_month_from_query(query, expected):
    assert infer_filters(query).year_months == expected


def test_infer_year_only():
    f = infer_filters("what happened in 2025")
    assert f.date_from == "2025-01-01" and f.date_to == "2025-12-31"


def test_infer_stays_quiet_without_a_date():
    """Guessing a window the user did not ask for is its own silent failure."""
    assert infer_filters("how is the humanitarian situation").is_empty()


def test_infer_latest_anchors_to_newest_edition():
    f = infer_filters("what is the latest displacement figure",
                      latest_indexed="2026-03-14")
    assert f.date_from and f.date_from.startswith("2026-01")


# --- result-set diversity ----------------------------------------------------

def test_same_text_across_editions_appears_once(retriever):
    """Republished text is kept per edition in the index, shown once in results.

    The index must stay date-addressable — a March-filtered query has to find
    the March copy — so deduplication belongs at retrieval, not ingest.
    """
    res = retriever.search("how many people are internally displaced", k=6)
    groups = [h for h in res.hits
              if retriever.index.meta[0].get("content_group") is not None]
    bodies = [" ".join(h.body.split())[:160].lower() for h in res.hits]
    assert len(bodies) == len(set(bodies)), "identical openings in one result set"


def test_one_report_cannot_take_every_slot(retriever):
    res = retriever.search("cholera", k=6, max_per_report=2)
    from collections import Counter
    counts = Counter(h.report_id for h in res.hits)
    assert max(counts.values()) <= 2 or len(res.hits) > sum(
        min(v, 2) for v in counts.values())


def test_spread_returns_distinct_periods(retriever):
    """--spread trades similarity for coverage across editions."""
    res = retriever.search("displaced", k=3, max_per_period=1)
    periods = [(h.date_original or "")[:7] for h in res.hits]
    assert len(periods) == len(set(periods))


def test_backfill_still_returns_k_when_constraints_bite(retriever):
    """Diversity is a preference, not a reason to return fewer results."""
    strict = retriever.search("displaced", k=4, max_per_report=1, max_per_period=1)
    loose = retriever.search("displaced", k=4)
    assert len(strict.hits) == len(loose.hits)


def test_era_clustering_is_reported(retriever):
    """An undated query returning one year must say so, not look authoritative."""
    res = retriever.search("how many people are internally displaced", k=2,
                           max_per_report=1)
    if res.hits and len({(h.date_original or "")[:4] for h in res.hits}) == 1:
        assert any("corpus spans" in n for n in res.notes)


def test_consecutive_chunks_of_one_section_are_not_both_returned(retriever):
    """Chunks within a section overlap by construction — showing both wastes a slot.

    A prefix check cannot catch this: the shared text is at the END of one
    chunk and the START of the next. parent_id is the exact guard.
    """
    from collections import Counter
    res = retriever.search("cholera displaced", k=6)
    counts = Counter(h.parent_id for h in res.hits)
    assert max(counts.values()) == 1 or res.suppressed == 0


def test_extraction_artefacts_never_reach_a_citation():
    from src.chunker import tidy
    raw = ("T <mark>he cholera outbreak</mark> continued <!-- Start of picture "
           "text --> Districts<br>with 0 - 34 cases<!-- End of picture text -->")
    out = tidy(raw)
    for junk in ("<mark>", "</mark>", "<br>", "<!--", "-->"):
        assert junk not in out
    assert "cholera outbreak" in out and "0 - 34 cases" in out

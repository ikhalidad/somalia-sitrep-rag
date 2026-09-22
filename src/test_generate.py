"""
Generation tests with a scripted LLM, so the pipeline's guarantees are tested
rather than the model's mood.

The point of a fake here is not convenience. Every check in generate.py is a
claim about what the system does when the model MISBEHAVES — cites an extract
that does not exist, states a figure with no source, answers an undated
question from a multi-edition context. You cannot test that by asking a
well-behaved model nicely; you have to script the misbehaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embed import build_index  # noqa: E402
from src.generate import (  # noqa: E402
    GroundedAnswerer, build_prompt, format_context, verify_answer,
)
from src.retrieve import Filters, Retriever  # noqa: E402
from tests.test_retrieve import EDITIONS, HashingEmbedder, _corpus  # noqa: E402


class ScriptedLLM:
    """Returns whatever you hand it, and records what it was asked."""

    model = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system: str | None = None
        self.user: str | None = None

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        self.system, self.user = system, user
        return self.reply


GOOD = ("An estimated 3.8 million people remained internally displaced as of "
        "2025-03-14 [1]. The figure rose to 4.4 million as of 2026-03-14 [2].")
UNCITED = ("An estimated 3.8 million people remained internally displaced. "
           "This represents an increase on prior periods.")
BAD_MARKER = "An estimated 3.8 million people were displaced as of 2025-03-14 [9]."
REFUSAL = ("The provided extracts do not contain information on maternal "
           "mortality rates.")


@pytest.fixture(scope="module")
def retriever():
    index = build_index(_corpus(), HashingEmbedder())
    return Retriever(index, HashingEmbedder())


def _answerer(retriever, reply: str, **kw):
    llm = ScriptedLLM(reply)
    return GroundedAnswerer(retriever, llm, min_dense=kw.pop("min_dense", 0.0), **kw), llm


# -- verification -----------------------------------------------------------

def test_good_answer_passes_audit():
    v = verify_answer(GOOD, n_sources=3)
    assert v.ok
    assert v.cited_markers == [1, 2]
    assert not v.invalid_markers
    assert v.citation_coverage == 1.0
    assert "2025-03-14" in v.dates_mentioned


def test_uncited_figure_is_caught():
    """The failure that matters most: a quotable number with no source."""
    v = verify_answer(UNCITED, n_sources=3)
    assert not v.ok
    assert v.uncited_figure_sentences
    assert "3.8 million" in v.uncited_figure_sentences[0]


def test_citation_to_a_nonexistent_extract_is_caught():
    v = verify_answer(BAD_MARKER, n_sources=3)
    assert not v.ok
    assert v.invalid_markers == [9]


def test_refusal_is_not_penalised_for_having_no_citations():
    v = verify_answer(REFUSAL, n_sources=3)
    assert v.is_refusal
    assert v.ok, "a refusal has nothing to cite and must not be flagged"


# -- prompt -----------------------------------------------------------------

def test_context_is_numbered_and_dated(retriever):
    res = retriever.search("displaced", k=3)
    ctx = format_context(res.hits)
    assert "[1]" in ctx and "[2]" in ctx
    for h in res.hits:
        assert h.date_original[:10] in ctx


def test_multi_period_context_warns_the_model(retriever):
    res = retriever.search("how many people are internally displaced", k=6)
    assert res.spans_periods
    system, _ = build_prompt("how many are displaced", res)
    assert "reporting periods" in system
    for p in res.periods:
        assert p in system


def test_single_period_context_carries_no_warning(retriever):
    res = retriever.search("cholera", k=3, filters=Filters(year_months=["2025-03"]))
    assert not res.spans_periods
    system, _ = build_prompt("cholera cases", res)
    assert "reporting periods" not in system


# -- abstain gate -----------------------------------------------------------

def test_abstains_when_filter_matches_nothing(retriever):
    a, llm = _answerer(retriever, GOOD)
    ans = a.answer("displacement in July 2019", filters=Filters(year_months=["2019-07"]))
    assert ans.abstained
    assert llm.user is None, "the model must not be called when abstaining"


def test_abstains_on_weak_match_rather_than_answering(retriever):
    """Out-of-corpus questions are the first thing a demo audience tries."""
    a, llm = _answerer(retriever, GOOD, min_dense=0.99)
    ans = a.answer("what is the exchange rate of the Somali shilling")
    assert ans.abstained
    assert "do not appear to cover" in ans.text
    assert llm.user is None


def test_answers_when_the_context_is_good(retriever):
    a, _ = _answerer(retriever, GOOD)
    ans = a.answer("how many people are internally displaced")
    assert not ans.abstained
    assert ans.trustworthy
    assert ans.flags == []


# -- end-to-end flagging ----------------------------------------------------

def test_uncited_figures_flagged_end_to_end(retriever):
    a, _ = _answerer(retriever, UNCITED)
    ans = a.answer("how many people are internally displaced")
    assert not ans.trustworthy
    assert any(f.startswith("uncited_figures") for f in ans.flags)


def test_undated_answer_from_multi_edition_context_is_flagged(retriever):
    """The project's thesis, enforced at the last mile.

    A model handed three editions that answers with one bare number has
    silently chosen an edition. The answer looks clean and may even be
    faithful to its chunk — which is exactly why a deterministic check has
    to catch it rather than an LLM judge.
    """
    a, _ = _answerer(retriever, UNCITED)
    ans = a.answer("how many people are internally displaced")
    assert ans.retrieval.spans_periods
    assert "undated_figures_across_periods" in ans.flags


def test_dated_answer_from_multi_edition_context_is_clean(retriever):
    a, _ = _answerer(retriever, GOOD)
    ans = a.answer("how many people are internally displaced")
    assert ans.retrieval.spans_periods
    assert "undated_figures_across_periods" not in ans.flags


def test_sources_record_which_extracts_were_actually_used(retriever):
    a, _ = _answerer(retriever, GOOD)
    ans = a.answer("how many people are internally displaced")
    cited = [s for s in ans.sources if s["cited"]]
    assert {s["n"] for s in cited} == {1, 2}
    assert all(s["url"] and s["date"] for s in ans.sources)


def test_dated_question_narrows_retrieval_automatically(retriever):
    a, _ = _answerer(retriever, GOOD)
    ans = a.answer("how many people were displaced in March 2026")
    assert ans.retrieval.filters_applied.year_months == ["2026-03"]
    assert all(h.date_original.startswith("2026-03") for h in ans.retrieval.hits)


def test_ask_cli_help_runs():
    """The ask CLI must import cleanly without a key or an index present."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "src.ask", "--help"],
                       cwd=root, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "--list-models" in r.stdout


def test_missing_api_key_explains_itself(monkeypatch, capsys):
    import src.ask as ask

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert ask.main(["anything"]) == 2
    assert "console.groq.com" in capsys.readouterr().err


@pytest.mark.parametrize("text,expected", [
    ("Reported in week 10 (4 - 10 March 2019) [1].", "10 March 2019"),
    ("An estimated 3.8 million as of March 2025 [1].", "March 2025"),
    ("Figures as of 2019-06-02 [1].", "2019-06-02"),
])
def test_full_dates_are_captured_not_bare_months(text, expected):
    """The year sat outside the capture group, so the audit under-reported itself."""
    assert expected in verify_answer(text, 1).dates_mentioned

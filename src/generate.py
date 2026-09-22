"""
Stage 6: grounded answer generation with verified citations.

The load-bearing idea here is that the model is not trusted to have followed
its instructions. Everything the prompt asks for — a citation on every claim,
a date on every figure, a refusal when the context is thin — is checked
programmatically afterwards, against the actual retrieved context. The prompt
is the request; `verify_answer` is the audit.

That matters because of how RAG evaluation usually goes wrong on this corpus.
Ragas `faithfulness` asks whether the answer follows from the retrieved
context, and on a serial publication it will happily score 1.0 for an answer
that quotes a 2019 IDP figure at a question about 2026 — the answer *does*
follow from the context it was given. The defect is that the figure is
undated and the wrong edition was retrieved. A cheap deterministic check
catches that before any LLM-as-judge metric is computed, and unlike the judge
it costs nothing per run and cannot itself hallucinate.

Three gates, in order:

    abstain  -> is the context good enough to answer from at all?
    generate -> answer under explicit, corpus-specific constraints
    verify   -> did it actually do what it was told, measured against context

Failing the third does not silently pass. `Answer.flags` travels with the
response and surfaces in the UI, because a grounded system that quietly emits
an uncited number is worse than one that visibly admits it did.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .chunker import tidy
from .retrieve import (
    Filters, Hit, RetrievalResult, Retriever, infer_filters,
)

log = logging.getLogger("generate")

# Below this cosine, the best match is not a match. RRF rank cannot express
# this: it reports the best of whatever survived the filter even when nothing
# survived that was any good.
#
# 0.45 is a starting value for bge-small-en-v1.5 with its query instruction,
# NOT a finding. Calibrate it on real data once the corpus lands: run the eval
# question set plus a set of deliberately off-topic questions, and put the
# threshold where the distributions separate. Record the number you chose and
# the plot in the README — an abstain threshold nobody calibrated is just a
# magic constant with good PR.
MIN_DENSE_SCORE = 0.45
MIN_HITS = 1


SYSTEM_PROMPT = """You answer questions about Somalia using ONLY the numbered \
extracts from humanitarian situation reports provided below.

Rules:

1. CITE EVERYTHING. Every factual claim ends with a marker in SQUARE BRACKETS \
naming the extract it came from: [1], [3], or [1][4] for several. Use ASCII \
square brackets, not any other bracket character. A sentence or bullet with a \
fact and no marker is a failure.

2. DATE EVERY FIGURE. Situation reports are serial — the same indicator is \
republished every edition with a different value. Write "as of <date>" using \
the date shown on the extract you took it from. An undated figure is wrong \
even when the number is right.

3. REPORT DISAGREEMENT, DO NOT RESOLVE IT. If extracts from different dates \
give different values for the same indicator, state both with their dates and \
say the figure changed. Never silently pick one.

4. REFUSE RATHER THAN GUESS. If the extracts do not contain the answer, say \
plainly what is missing and stop. Do not use knowledge from outside the \
extracts. Do not estimate, interpolate, or reason from what is typical.

5. NO PREAMBLE. Answer the question directly. Do not restate it. Do not \
describe the extracts."""

MULTI_PERIOD_WARNING = """
NOTE: the extracts below span {n} different reporting periods ({periods}). \
Any figure you report MUST carry the date of the extract it came from, and \
where values differ across periods you must say so."""


# --------------------------------------------------------------------------
# provider abstraction
# --------------------------------------------------------------------------

class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str: ...


class OpenAICompatLLM:
    """Any OpenAI-shaped /chat/completions endpoint.

    One class covers Groq, Together, OpenRouter, DeepInfra, Fireworks and a
    local vLLM or Ollama server, because they all speak the same wire format.
    The provider is a base_url and a key, not a code change — which is what
    you want when the deployment target is a free tier whose terms may move.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        base_url: str | None = None,
        api_key_env: str = "LLM_API_KEY",
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI  # lazy: keeps the import off the test path

        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise ValueError(f"{api_key_env} is not set")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        # Temperature 0. This is an extraction task with a right answer, and
        # sampling variance would make the eval harness measure the sampler
        # rather than the pipeline.
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        import random
        import time

        # Free tiers limit requests per minute. Interactive use rarely notices;
        # the eval harness walks straight into it, and a hard failure there
        # loses a whole run's results.
        delay = 2.0
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                retriable = ("rate" in msg or "429" in msg or "timeout" in msg
                             or "overload" in msg or "503" in msg)
                if not retriable or attempt == 4:
                    raise
                wait = delay + random.random()
                log.warning("%s — retrying in %.0fs", type(exc).__name__, wait)
                time.sleep(wait)
                delay *= 2
        raise RuntimeError("unreachable")


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key_env: str = "ANTHROPIC_API_KEY",
        temperature: float = 0.0,
    ) -> None:
        import anthropic  # lazy

        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise ValueError(f"{api_key_env} is not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            temperature=self.temperature,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text").strip()


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

def format_context(hits: Sequence[Hit], use_parents: bool = True) -> str:
    """Numbered extracts. The number is the citation key the model must use."""
    blocks = []
    for n, h in enumerate(hits, start=1):
        date = (h.date_original or "undated")[:10]
        src = "/".join(h.sources or []) or "ReliefWeb"
        sector = f" | {h.sector}" if h.sector else ""
        body = tidy(h.parent_text if use_parents and h.parent_text else h.body)
        blocks.append(
            f"[{n}] {date} | {src}{sector}\n"
            f"{h.title or 'Situation Report'}\n"
            f"{body.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, result: RetrievalResult,
                 use_parents: bool = True) -> tuple[str, str]:
    system = SYSTEM_PROMPT
    if result.spans_periods:
        system += MULTI_PERIOD_WARNING.format(
            n=len(result.periods), periods=", ".join(result.periods)
        )
    user = (f"EXTRACTS\n\n{format_context(result.hits, use_parents)}\n\n"
            f"---\n\nQUESTION: {question}")
    return system, user


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

# Models emit citation markers in whichever bracket their tokenizer favours.
# gpt-oss returned 【1】 on a correctly-cited answer, which scored 0% coverage
# against a regex that only knew [1] — the audit failing, not the model.
_MARKER_RE = re.compile(r"[\[【［]\s*(\d{1,2})\s*[\]】］]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_MONTHS = (r"january|february|march|april|may|june|july|august|september"
           r"|october|november|december")
# The year sat OUTSIDE the capture group, so "March 2019" was recorded as
# "March" and the audit under-reported its own result: an answer that dated
# every figure correctly looked like one that named bare months.
_DATE_IN_ANSWER = re.compile(
    rf"\b(20[0-3]\d-\d{{2}}(?:-\d{{2}})?)\b"
    rf"|\b((?:\d{{1,2}}\s+)?(?:{_MONTHS})\s+20[0-3]\d)\b", re.I)
# Same register problem as the chunker: "per cent" and "billion", not just "%".
_FIGURE_IN_ANSWER = re.compile(
    r"\b\d[\d,.]*\s*(?:per\s?cent|percent|%|billion|million|thousand"
    r"|people|persons|cases|children|women|households|idps?|usd|dollars?)"
    r"|\$\s?\d|\b\d{1,3}(?:,\d{3})+\b", re.I)


@dataclass
class Verification:
    cited_markers: list[int] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    uncited_figure_sentences: list[str] = field(default_factory=list)
    citation_coverage: float = 0.0       # share of substantive sentences cited
    dates_mentioned: list[str] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def ok(self) -> bool:
        return (not self.invalid_markers
                and not self.uncited_figure_sentences
                and (self.is_refusal or self.citation_coverage >= 0.6))


# Matches a statement ABOUT the extracts rather than a claim drawn from them.
# Used twice: to recognise an outright refusal, and to exclude honest
# absence-statements from the citation-coverage denominator. It missed "do not
# provide", so the caveat we most want an answer to include — naming what the
# sources cannot support — was scored as an uncited claim.
_REFUSAL_RE = re.compile(
    r"\b(?:do(?:es)?n?'?t? ?(?:not)? ?(?:contain|provide|include|give|specify"
    r"|report|state|mention|break down|disaggregate)"
    r"|not (?:available|present|found|included|reported|specified|in)"
    r"|no (?:information|data|figures?|extracts?|breakdown|separate)"
    r"|cannot (?:answer|determine|find|be determined)|unable to"
    r"|only give|only provide|only report)\b", re.I)


# LLM output is full of typographic punctuation. gpt-oss wrote "2019‑03‑10"
# with U+2011 NON-BREAKING HYPHEN, so a date regex expecting ASCII "-" reported
# "dates stated: none" for an answer that dated every figure. Normalise before
# any pattern touches the text.
_UNICODE_PUNCT = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u00a0": " ",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
})


def normalize_punct(text: str) -> str:
    return text.translate(_UNICODE_PUNCT)


def verify_answer(text: str, n_sources: int) -> Verification:
    """Audit the generated answer against what the prompt demanded.

    Deterministic and free. It runs before any LLM-as-judge metric and
    catches the two failures a judge is worst at spotting: a citation
    pointing at an extract that was never supplied, and a figure asserted
    with no source attached.
    """
    v = Verification()
    text = normalize_punct(text)
    markers = [int(m) for m in _MARKER_RE.findall(text)]
    v.cited_markers = sorted(set(markers))
    v.invalid_markers = sorted({m for m in markers if not 1 <= m <= n_sources})
    v.dates_mentioned = sorted({d for grp in _DATE_IN_ANSWER.findall(text)
                                for d in grp if d})
    v.is_refusal = bool(_REFUSAL_RE.search(text)) and not markers

    units = _claim_units(text)
    # A statement about the extracts themselves — "the extracts do not provide
    # a cumulative total" — is the honest behaviour the prompt asked for and
    # has nothing to cite. Counting it as an uncited claim penalises exactly
    # the answer we want.
    claims = [u for u in units if not _REFUSAL_RE.search(u)]
    if claims:
        cited = sum(1 for u in claims if _MARKER_RE.search(u))
        v.citation_coverage = round(cited / len(claims), 3)

    # A number without a marker is the failure mode that matters most: it is
    # the most quotable part of the answer and the least verifiable.
    v.uncited_figure_sentences = [
        u for u in claims
        if _FIGURE_IN_ANSWER.search(u) and not _MARKER_RE.search(u)
    ]
    return v


_BULLET_RE = re.compile(r"^\s*(?:[-*•‣]|\d+[.)])\s+")


def _claim_units(text: str) -> list[str]:
    """Split an answer into units that each need their own citation.

    Sentence splitting alone is wrong for this output. The natural shape of a
    multi-edition answer is a bullet per edition, and the sentence rule —
    a full stop followed by a capital — never fires between "…(8-14 April
    2019) [5]." and "- 41 new cases…", so six cited bullets counted as one
    unit and the coverage figure was meaningless.
    """
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _BULLET_RE.match(line):
            units.append(line)
        else:
            units.extend(p.strip() for p in _SENTENCE_RE.split(line) if p.strip())
    return [u for u in units if len(u.split()) > 4]


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict[str, Any]]        # one per cited extract, for the UI
    retrieval: RetrievalResult | None
    verification: Verification | None
    abstained: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return bool(self.verification and self.verification.ok and not self.flags)


class GroundedAnswerer:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        k: int = 6,
        min_dense: float = MIN_DENSE_SCORE,
        use_parents: bool = True,
        auto_filters: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_dense = min_dense
        self.use_parents = use_parents
        self.auto_filters = auto_filters
        self._latest = max(
            ((m.get("date_original") or "")[:10] for m in retriever.index.meta),
            default=None,
        )

    def answer(self, question: str, filters: Filters | None = None) -> Answer:
        if filters is None and self.auto_filters:
            filters = infer_filters(question, latest_indexed=self._latest)

        result = self.retriever.search(
            question, k=self.k, filters=filters, expand_parents=self.use_parents
        )

        refusal = self._abstain_reason(result)
        if refusal:
            return Answer(question, refusal, [], result, None, abstained=True,
                          flags=["abstained"])

        system, user = build_prompt(question, result, self.use_parents)
        text = self.llm.complete(system, user)

        v = verify_answer(text, len(result.hits))
        flags: list[str] = []
        if v.invalid_markers:
            flags.append(f"invalid_citations:{v.invalid_markers}")
        if v.uncited_figure_sentences:
            flags.append(f"uncited_figures:{len(v.uncited_figure_sentences)}")
        if not v.is_refusal and v.citation_coverage < 0.6:
            flags.append(f"low_citation_coverage:{v.citation_coverage}")
        # The whole point of the temporal work: a multi-edition context that
        # produces an answer naming no date has silently picked one edition.
        if result.spans_periods and not v.dates_mentioned and not v.is_refusal:
            flags.append("undated_figures_across_periods")

        return Answer(question, text, self._sources(result.hits, v.cited_markers),
                      result, v, flags=flags)

    def _abstain_reason(self, result: RetrievalResult) -> str | None:
        """Refuse rather than answer from a context that cannot support one.

        A RAG system that always answers is a RAG system that hallucinates on
        out-of-corpus questions, and those are exactly the questions a
        demo audience will try first.
        """
        if not result.hits:
            note = result.notes[0] if result.notes else ""
            return ("No indexed situation reports match that question. "
                    + note).strip()
        if len(result.hits) < MIN_HITS:
            return "Too few matching extracts to answer reliably."
        if result.max_dense < self.min_dense:
            return (
                "The indexed situation reports do not appear to cover that "
                f"question (best match scored {result.max_dense:.2f}, below the "
                f"{self.min_dense:.2f} threshold). Try rephrasing, or ask about "
                "a topic the reports cover: displacement, health, nutrition, "
                "WASH, protection, food security or funding."
            )
        return None

    @staticmethod
    def _sources(hits: Sequence[Hit], cited: Sequence[int]) -> list[dict[str, Any]]:
        out = []
        for n, h in enumerate(hits, start=1):
            out.append({
                "n": n, "cited": n in cited, "citation": h.citation(),
                "url": h.url, "date": (h.date_original or "")[:10],
                "sector": h.sector, "sources": h.sources,
                "dense_score": h.dense_score, "chunk_id": h.chunk_id,
            })
        return out

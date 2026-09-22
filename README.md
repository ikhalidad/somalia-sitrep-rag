# Somalia Situation Report RAG Assistant

Question answering over 900 ReliefWeb situation reports on Somalia (2018–2026),
with source citations, date-aware retrieval, and a programmatic audit of every
answer the model produces.

---

## What it does

```
$ python -m src.ask "how many cholera cases were reported in Banadir?"

- 32 new cholera cases were reported in Banadir CTC in epidemiological week 10
  (4 – 10 March 2019) [6].
- 30 new cholera cases in epidemiological week 15 (8 – 14 April 2019) [5].
- 41 new cholera cases in epidemiological week 16 (15 – 21 April 2019) [3].
- 47 new cholera cases in epidemiological week 18 (28 April – 5 May 2019) [2].
- 133 new cholera cases in epidemiological week 22 (27 May – 2 June 2019) [1].
- 149 new cholera cases in epidemiological week 23 (3 – 9 June 2019) [4].

The extracts provide cumulative totals for all five regions (e.g. 7,501 cases
as of 2 June 2019) but do not give a separate cumulative count for Banadir alone.

*[1] WHO/Govt. Somalia, 2019-06-02 — AWD/Cholera Epi Week 22  (reliefweb.int/node/3186189)
*[2] WHO/Govt. Somalia, 2019-05-05 — AWD/Cholera Epi Week 18  (reliefweb.int/node/3186184)
 ... four more, all cited

citation coverage 100%  |  dates stated: 10 March 2019, 2 June 2019, ...
verified: every claim cited, every figure dated
```

Two things there matter more than the answer itself.

It returned a **series, not a number**. Asked an undated question against six
editions of a serial publication, it reported all six with their dates rather
than picking one and sounding certain.

It **named what the sources do not support**. The cumulative Banadir total is
genuinely absent from the retrieved extracts, and it said so rather than
computing something plausible.

---

## The problem this corpus poses

Most RAG demonstrations index unrelated documents. Situation reports are
**serial**: the same skeleton — highlights, key figures, a block per cluster,
funding — refilled every fortnight with new numbers. Across eight years that
produces thousands of documents that are, section for section, near-copies.

### Temporal collision

> "An estimated 3.8 million people remain internally displaced."

That sentence appears in sixty editions with sixty different figures. As
embeddings they are nearly indistinguishable — cosine above 0.97. Ask *"how
many people are displaced in Somalia?"* and the retriever returns a
semantically arbitrary edition. It might be from 2019. The model then answers
fluently, cites correctly, and is wrong by several years.

**This is why the project exists.** Note what it does to evaluation: Ragas
`faithfulness` scores that answer 1.0, because the answer *is* supported by the
retrieved context. The defect is upstream, in retrieval, and only
`context_precision` against date-aware ground truth catches it.

Four mechanisms, none retrofittable at query time:

1. The publication date is written **into the embedded text**, so the vector
   itself carries temporal signal.
2. Date, sector and region are **filterable metadata** — a March 2025 question
   is hard-filtered to March 2025 *before* scoring rather than after. The naive
   order returns nothing from a corpus that holds the answer.
3. Every chunk is **self-dating**, so the generator can be required to state an
   as-of date and will visibly fail rather than silently mislead.
4. When results cluster in one era, the system **says so**: *"all results fall
   in 2019-03..2019-06, but the corpus spans 2018-01..2026-09."*

---

## What the data taught us

Every decision below began as an assumption and was overturned by running
`python -m src.ingest profile` against the real corpus.

**40% of documents tagged "Somalia" are not about Somalia.** A query for
Somalia situation reports returns 2,275 documents; only 902 have Somalia as
their `primary_country`. Afghanistan alone accounts for 462 — global UNHCR and
WHO products tagged with every country whose refugees they count. Filtering on
`country` rather than `primary_country` more than doubles the corpus with
documents about Yemen, Ukraine and Libya.

**ReliefWeb's `body` field is 1% structured; the PDFs are 96%.** The first
resolution rule preferred `body` whenever it was long enough — it is clean
Markdown with no extraction risk. The profile showed body-sourced documents
yielding a median of **one** section against **eleven** for PDF-sourced. The
body is a prose rendering with the headings flattened out. Length was the wrong
test; resolution now prefers whichever source carries the structure.

**WHO is half this corpus. OCHA is five percent.** The chunker's section
vocabulary was modelled on OCHA's cluster template — Highlights, Key Figures,
Health, WASH, Protection. It matched 4% of real sections. WHO publishes EWARN
epidemiological bulletins (surveillance, laboratory, proportional morbidity,
case management); UNHCR publishes operational updates (new arrivals, cash
assistance, self-reliance). Rebuilding the vocabulary from headings the corpus
actually produced took matching from 4% to 73%.

**Reports organise by region as often as by sector.** Regions appear as section
headings — "Banadir region", "Hiraan region" — and region is the commonest
qualifier in real questions. All 18 admin1 regions are a retrieval facet, with
major towns mapped to their region and Federal Member States expanded to their
constituent regions.

Sool and Sanaag are deliberately assigned to **neither** Puntland nor
Somaliland. Their administration is contested, and encoding either claim in a
retrieval facet is a political position this system has no business taking.

---

## Pipeline

```
config/corpus.yaml                 the corpus is defined once, here
        │
        ▼
[1] ingest sources    1 API call — which publishers exist, before fetching
[2] ingest fetch      metadata + PDF attachments, resumable, quota-tracked
[3] ingest resolve    body vs PDF, whichever carries the structure
[4] ingest profile    corpus shape + structure analysis (no API calls)
        │
        ▼
[5] chunker           section-aware chunks with contextual headers
[6] embed             bge-small-en-v1.5, flat float32 index, BM25 sidecar
        │
        ▼
[7] search            retrieval only — no LLM, no API key
[8] ask               grounded answer + citation audit
```

### Design notes

**No vector database.** ~20,000 chunks at 384 dimensions in float32 is 30 MB —
small enough to hold in RAM and scan exactly in milliseconds. An ANN index
trades recall for speed at a scale this corpus does not reach; adding one buys
approximation error and a startup dependency for no measurable latency win.

**Hybrid retrieval, fused on rank.** Dense embeddings are weakest on exactly
what this corpus is made of: acronyms (PRMN, IPC, CFR, SAM/MAM), Somali place
names, and numerals. BM25 is strongest there. Reciprocal Rank Fusion combines
them on rank rather than score, because cosine and BM25 live on unrelated
scales and any interpolation constant is indefensible.

**Tables are never split mid-row.** A table too large for one chunk is split at
row boundaries with its header repeated in each piece, so every row keeps the
column labels that give its numbers meaning. Emitting oversized tables whole
was worse: the embedding model truncates at its window silently, so the tail
was never embedded while still appearing in the text.

**The model is not trusted to have followed instructions.** Everything the
prompt demands — a citation per claim, a date per figure, a refusal when the
context is thin — is checked afterwards against the retrieved context.
`verify_answer` is deterministic, free, cannot itself hallucinate, and runs
before any LLM-as-judge metric. It catches the two things a judge is worst at:
a citation pointing at an extract that was never supplied, and a figure
asserted with no source attached. Failures travel with the answer as flags.

**It refuses.** Out-of-corpus questions abstain on raw cosine before the model
is called at all. A RAG system that always answers is one that hallucinates on
the first question a demo audience tries.

---

## Running it

Requires a [ReliefWeb appname](https://apidoc.reliefweb.int/) (free, mandatory
since 1 November 2025) and any OpenAI-compatible LLM key.

```bash
python -m venv .venv && source .venv/bin/activate
# CPU-only torch first, or pip pulls ~3 GB of CUDA wheels you do not need
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export RELIEFWEB_APPNAME='yourorg-purpose-random'
python -m src.ingest sources      # inspect publishers before fetching
python -m src.ingest fetch        # ~30 API calls, then PDF downloads
python -m src.ingest resolve      # PDF → Markdown, parallel and cached
python -m src.ingest profile      # corpus shape and structure

python -m src.chunker
python -m src.embed               # ~100 min on 12 CPU cores, resumable

python -m src.search "cholera cases in Banadir"       # no LLM needed
export LLM_API_KEY=...
python -m src.ask "how many cholera cases in Banadir?"
```

Every long step is resumable. Raw API responses, PDFs and extracted text are
cached on disk, so re-chunking and re-embedding never touch the network — which
matters when the budget is 1,000 API calls a day. Downloads are written
atomically: a file only exists once complete, so an interrupted run cannot
leave a truncated PDF that later looks cached.

```bash
pytest tests/ -q     # 132 tests, no network, no model download
```

The suite runs against a synthetic sitrep fixture with a deterministic
stand-in embedder and a scripted LLM. That is deliberate: the guarantees worth
testing are what the system does when the model **misbehaves** — cites an
extract that does not exist, states a figure with no source, answers an undated
question from a multi-edition context. You cannot test that by asking a
well-behaved model nicely.

---

## Data governance

ReliefWeb content is contributed by partner organisations and may carry
copyright held by the original source. This repository ships **the code that
reproduces the corpus, not the corpus** — `data/` is excluded. Answers quote
short passages with an attributed link back to the ReliefWeb record; nothing is
republished wholesale. All indexed material is already public: no
protection-sensitive or personally identifiable data enters the pipeline.

---

## Status

Working end to end: ingestion, chunking, indexing, retrieval, grounded
generation, 132 tests.

Next: a Ragas evaluation harness with date-aware ground truth — including the
comparison that makes the case, naive fixed-size chunking against this pipeline
scored on context precision — and a Hugging Face Space deployment.

Known gaps, stated plainly:

- `MIN_DENSE_SCORE = 0.45`, the abstain threshold, is a starting value and has
  not been calibrated against a set of deliberately out-of-corpus questions.
- 27% of sections still carry no sector facet — mostly document titles and
  long-tail headings. Section *boundaries* are at 97%; only the facet is
  incomplete.
- Food security is absent by construction: FEWS NET, WFP and FAO were excluded
  when the corpus was scoped to three publishers. Whether to widen is a
  question for the evaluation harness, not a guess.

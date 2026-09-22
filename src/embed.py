"""
Stage 4: embed chunks and build the retrieval index.

Why there is no vector database here
------------------------------------
This corpus is on the order of 3,000-8,000 chunks. At 384 dimensions in
float32 that is roughly 5-12 MB — small enough to hold in RAM, small enough
to commit to the Space repo, small enough that exact cosine over the whole
matrix returns in single-digit milliseconds.

An approximate-nearest-neighbour index exists to trade recall for speed at a
scale this corpus does not reach. Adding Chroma or FAISS here would mean
accepting approximation error, a persistence layer, and a startup dependency
on a free-tier Space with ephemeral disk — in exchange for no measurable
latency win. One numpy matmul is the correct engineering answer, and saying
so with the arithmetic attached is a better interview answer than naming a
vector database because tutorials name one.

The threshold to revisit: around 10^6 chunks, or when the index stops fitting
in Space RAM. `INDEX_FORMAT_NOTE` in this module records that so the decision
is legible to whoever reads the repo next.

Hybrid retrieval
----------------
Dense embeddings are weakest on exactly what this corpus is made of:
acronyms (PRMN, IPC, CFR, HRP, SAM/MAM), Somali place names (Banadir, Bay,
Bakool, Gedo, Hiraan, Middle Juba), and numerals. `sentence-transformers`
smears "Gedo" toward every other region; BM25 does not. So both indexes are
built here and fused at query time in retrieve.py.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

log = logging.getLogger("embed")

INDEX_FORMAT_NOTE = (
    "Flat float32 matrix, exact cosine. Revisit ANN (hnswlib/FAISS) above "
    "~10^6 chunks or when the matrix stops fitting comfortably in Space RAM."
)

# Somali place names, cluster acronyms and humanitarian jargon that a general
# tokenizer mangles. Kept lowercase; used to protect tokens during BM25
# normalisation so "SAM" does not collapse into "sam".
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/']*")


class Embedder(Protocol):
    """Anything that turns strings into unit-norm row vectors.

    Declared as a Protocol rather than hardcoding SentenceTransformer so the
    test suite can inject a deterministic stand-in and run with no model
    download and no network.
    """

    dim: int

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """bge-small-en-v1.5 by default. CPU-friendly, 512-token limit, 384 dims."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        import os

        import torch
        from sentence_transformers import SentenceTransformer  # lazy

        # WSL and container runtimes frequently hand torch a single thread,
        # which turns a ten-minute embedding run into ninety. Ask for every
        # core available before the model is built.
        cores = os.cpu_count() or 1
        if torch.get_num_threads() < cores:
            torch.set_num_threads(cores)
        log.info("torch using %d of %d cores", torch.get_num_threads(), cores)

        self.model = SentenceTransformer(model_name, device=device)
        # Explicit: a chunk longer than this is truncated with no error, so
        # the number must match the budget the chunker sized against.
        self.model.max_seq_length = 512
        self.batch_size = batch_size
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or self.model.get_sentence_embedding_dimension)
        self.dim = get_dim()
        self.model_name = model_name
        # BGE models are trained with an asymmetric query instruction. Omitting
        # it costs a few points of retrieval quality — a silent, free loss that
        # is easy to miss because nothing errors.
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower() else ""
        )

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,   # so cosine == dot product
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


def tokenize(text: str) -> list[str]:
    """Lexical tokenizer for BM25. Numbers kept — they are the payload here."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ChunkIndex:
    """Everything the retriever needs, serialisable to a directory."""

    vectors: np.ndarray            # (n, d) float32, L2-normalised
    meta: list[dict[str, Any]]     # per-chunk metadata, aligned to vectors
    model_name: str
    bm25: Any = None               # rank_bm25.BM25Okapi or None

    def __len__(self) -> int:
        return len(self.meta)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with (d / "meta.jsonl").open("w", encoding="utf-8") as fh:
            for m in self.meta:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        (d / "manifest.json").write_text(json.dumps({
            "model_name": self.model_name,
            "n_chunks": len(self.meta),
            "dim": int(self.vectors.shape[1]),
            "dtype": str(self.vectors.dtype),
            "megabytes": round(self.vectors.nbytes / 1e6, 2),
            "has_bm25": self.bm25 is not None,
            "format_note": INDEX_FORMAT_NOTE,
        }, indent=2))
        if self.bm25 is not None:
            with (d / "bm25.pkl").open("wb") as fh:
                pickle.dump(self.bm25, fh)
        log.info("index saved to %s (%.1f MB)", d, self.vectors.nbytes / 1e6)

    @classmethod
    def load(cls, path: str | Path) -> "ChunkIndex":
        d = Path(path)
        manifest = json.loads((d / "manifest.json").read_text())
        meta = [json.loads(l) for l in
                (d / "meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        bm25 = None
        if (d / "bm25.pkl").exists():
            with (d / "bm25.pkl").open("rb") as fh:
                bm25 = pickle.load(fh)
        return cls(np.load(d / "vectors.npy"), meta, manifest["model_name"], bm25)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

# Fields carried into the index. Everything here is filterable or citable at
# query time; anything not listed is dead weight in Space RAM.
META_FIELDS = (
    "chunk_id", "report_id", "parent_id", "body", "header", "date_original",
    "year_month", "sources", "sector", "regions", "heading", "title", "url",
    "text_source", "is_primary_country_som", "n_tokens", "contains_table",
    "contains_figures", "content_group", "n_editions",
)


def build_index(
    chunks: Iterable[dict[str, Any]],
    embedder: Embedder,
    with_bm25: bool = True,
    shard_dir: str | Path | None = None,
    shard_size: int = 2048,
) -> ChunkIndex:
    chunks = list(chunks)
    if not chunks:
        raise ValueError("no chunks to index — run src.chunker first")

    texts = [c["text"] for c in chunks]
    log.info("embedding %d chunks with %s", len(texts),
             getattr(embedder, "model_name", type(embedder).__name__))
    vectors = (_encode_sharded(texts, embedder, Path(shard_dir), shard_size)
               if shard_dir else embedder.encode(texts, is_query=False))

    # Guard against a silently un-normalised embedder: cosine-as-dot-product
    # is only valid on unit vectors, and the failure is invisible — scores
    # stay plausible, ranking quietly degrades.
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        log.warning("vectors not unit-norm (mean %.3f) — normalising", norms.mean())
        vectors = vectors / np.clip(norms, 1e-9, None)[:, None]

    meta = [{k: c.get(k) for k in META_FIELDS} for c in chunks]

    bm25 = None
    if with_bm25:
        try:
            from rank_bm25 import BM25Okapi

            # Indexed on body, not text: the contextual header is identical
            # across every chunk of a document, and letting BM25 score it
            # would reward the header rather than the content.
            bm25 = BM25Okapi([tokenize(c["body"]) for c in chunks])
        except ImportError:
            log.warning("rank_bm25 not installed — dense-only retrieval. "
                        "Expect weaker recall on acronyms and place names.")

    return ChunkIndex(vectors.astype(np.float32), meta,
                      getattr(embedder, "model_name", "unknown"), bm25)


def _encode_sharded(texts: list[str], embedder: Embedder, shard_dir: Path,
                    shard_size: int) -> np.ndarray:
    """Embed in shards, saving each as it completes.

    Embedding is the longest unattended step in the pipeline. Without
    checkpointing, a closed laptop at 80% costs the whole run — which is not
    a hypothetical failure mode on a machine that sleeps.

    Shards are keyed to the corpus: if the chunk set changes, every shard is
    discarded rather than silently mixing vectors from two different corpora.
    """
    import hashlib
    import time

    shard_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        f"{len(texts)}|{texts[0][:200]}|{texts[-1][:200]}".encode()
    ).hexdigest()[:16]
    stamp = shard_dir / "corpus.txt"
    if stamp.exists() and stamp.read_text().strip() != fingerprint:
        log.warning("chunk set changed — discarding %d stale shards",
                    len(list(shard_dir.glob("*.npy"))))
        for f in shard_dir.glob("*.npy"):
            f.unlink()
    stamp.write_text(fingerprint)

    bounds = list(range(0, len(texts), shard_size))
    parts: list[np.ndarray] = []
    t0 = time.monotonic()
    done_now = 0
    for n, start in enumerate(bounds):
        path = shard_dir / f"{n:05d}.npy"
        if path.exists():
            parts.append(np.load(path))
            continue
        part = embedder.encode(texts[start:start + shard_size], is_query=False)
        np.save(path, part.astype(np.float32))
        parts.append(part)
        done_now += 1
        rate = done_now / max(time.monotonic() - t0, 1e-6)
        left = (len(bounds) - n - 1) / max(rate, 1e-9)
        log.info("shard %d/%d saved  ~%.0f min left", n + 1, len(bounds), left / 60)
    return np.vstack(parts)


def run(cfg: dict[str, Any], embedder: Embedder | None = None) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed"])
    chunks = [json.loads(l) for l in
              (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    embedder = embedder or SentenceTransformerEmbedder(
        cfg["chunking"].get("embedding_model", "BAAI/bge-small-en-v1.5")
    )
    index = build_index(chunks, embedder, shard_dir=processed / "index" / "shards")
    index.save(processed / "index")
    return json.loads((processed / "index" / "manifest.json").read_text())


if __name__ == "__main__":
    import argparse

    from .ingest import load_config, quiet_logs

    ap = argparse.ArgumentParser(prog="src.embed")
    ap.add_argument("--config", default="config/corpus.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    quiet_logs()
    print(json.dumps(run(load_config(a.config)), indent=2))

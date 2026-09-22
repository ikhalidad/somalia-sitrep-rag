"""Chunker tests against a synthetic sitrep shaped like the real thing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunker import (  # noqa: E402
    SitrepChunker, build_header, classify_sector, split_blocks, split_sections,
)

CFG = {
    "chunking": {
        "embedding_model": None,  # heuristic counter; CI has no HF download
        "max_tokens": 512, "header_reserve": 60, "target_tokens": 400,
        "overlap_tokens": 64, "min_tokens": 48,
        "atomic_tables": True, "max_table_tokens": 1200,
        "emit_parent_sections": True,
    },
    "dedup": {"exact_hash": True, "boilerplate_doc_threshold": 2,
              "boilerplate_requires_no_figures": True, "drop_boilerplate": True},
}

SITREP = """
# Somalia: Humanitarian Situation Report

## HIGHLIGHTS

- An estimated 3.8 million people remain internally displaced across Somalia,
  the majority in Banadir, Bay and Lower Shabelle regions.
- Drought conditions persisted through the Jilaal season, with below-average
  Deyr rainfall recorded in 14 of 18 regions.
- Humanitarian partners reached 2.1 million people with some form of
  assistance during the reporting period.

## KEY FIGURES

| Indicator | Figure | Change |
| --- | --- | --- |
| People in need | 6.9 million | +0.4m |
| People targeted | 4.2 million | unchanged |
| People reached | 2.1 million | +0.2m |
| IDPs | 3.8 million | +0.1m |

## Health Cluster

Health partners supported 412 health facilities across 54 districts during the
reporting period. A total of 1,204 suspected cholera cases were reported from
Banadir, with 4 associated deaths, giving a case fatality rate of 0.3 per cent.
Measles vaccination campaigns reached 88,000 children under five in Bay and
Bakool. Partners report continued shortages of essential medicines in Gedo,
where three facilities suspended outpatient services during the period.
Referral pathways between primary facilities and regional hospitals remain
weak in Middle Juba, where insecurity restricted partner movement for eleven
days of the reporting period.

## Nutrition

An estimated 1.8 million children under five are projected to face acute
malnutrition through the end of the year, including 479,000 severe cases.
Nutrition partners admitted 62,400 children to therapeutic feeding programmes,
a 12 per cent increase on the previous period. Programme coverage remains
below target in Middle Juba and Hiraan.

## Protection

Protection partners documented 1,940 incidents during the reporting period.
Gender-based violence service points recorded 3,112 cases, of which 61 per cent
involved displaced women and girls. Child protection actors identified 890
unaccompanied and separated children, and reunified 214 with their families.
Case management capacity remains the binding constraint in Banadir.

## FUNDING

The Humanitarian Response Plan is 41 per cent funded against requirements of
1.6 billion United States dollars. Underfunded sectors include protection at
22 per cent and education at 18 per cent.

## About OCHA

OCHA coordinates the global emergency response to save lives and protect
people in humanitarian crises. We advocate for effective and principled
humanitarian action by all, for all.
"""

DOC = {
    "report_id": 4211987,
    "title": "Somalia: Humanitarian Situation Report No. 4 (March 2025)",
    "text": SITREP,
    "date_original": "2025-03-14T00:00:00+00:00",
    "sources": ["UN Office for the Coordination of Humanitarian Affairs"],
    "source_shortnames": ["OCHA"],
    "url": "https://reliefweb.int/report/somalia/example-4",
    "text_source": "body",
    "is_primary_country_som": True,
}


def test_sections_are_found_and_typed():
    secs = split_sections(SITREP)
    sectors = [s.sector for s in secs]
    assert "Highlights" in sectors
    assert "Key Figures" in sectors
    assert "Health" in sectors
    assert "Nutrition" in sectors
    assert "Protection" in sectors
    assert "Funding" in sectors


def test_sector_classification():
    assert classify_sector("Health Cluster") == "Health"
    assert classify_sector("WASH") == "WASH"
    assert classify_sector("Gender-Based Violence") == "Protection"
    assert classify_sector("Food Security and Livelihoods") == "Food Security"
    assert classify_sector("Mogadishu") is None  # place names are not sections


def test_table_stays_whole():
    figures = next(s for s in split_sections(SITREP) if s.sector == "Key Figures")
    blocks = split_blocks(figures.text)
    tables = [b for b, is_t in blocks if is_t]
    assert len(tables) == 1
    for row in ("People in need", "People targeted", "IDPs"):
        assert row in tables[0], "table was fragmented"


def test_no_chunk_crosses_a_sector_boundary():
    chunks = SitrepChunker(CFG).chunk_document(DOC)
    for c in chunks:
        if c.sector == "Health":
            assert "unaccompanied and separated children" not in c.body
            assert "therapeutic feeding" not in c.body


def test_header_carries_date_source_and_sector():
    secs = split_sections(SITREP)
    health = next(s for s in secs if s.sector == "Health")
    header = build_header(DOC, health)
    assert "2025-03-14" in header
    assert "OCHA" in header
    assert "Health" in header


def test_every_chunk_is_self_dating():
    """The temporal-collision mitigation, asserted."""
    for c in SitrepChunker(CFG).chunk_document(DOC):
        assert "2025-03-14" in c.text
        assert c.year_month == "2025-03"


def test_chunks_fit_the_embedding_budget():
    ch = SitrepChunker(CFG)
    for c in ch.chunk_document(DOC):
        if not c.contains_table:
            assert c.n_tokens <= CFG["chunking"]["max_tokens"], \
                f"{c.chunk_id} at {c.n_tokens} tokens will be silently truncated"


def test_boilerplate_detected_across_editions():
    """'About OCHA' appears in every edition — it must not reach the index."""
    ch = SitrepChunker(CFG)
    chunks = []
    for i, d in enumerate(("2025-01-14", "2025-02-14", "2025-03-14")):
        doc = {**DOC, "report_id": 4211900 + i, "date_original": f"{d}T00:00:00+00:00"}
        chunks.extend(ch.chunk_document(doc))
    ch.mark_duplicates(chunks)
    about = [c for c in chunks if "coordinates the global emergency response" in c.body]
    assert about, "fixture changed"
    assert any(c.is_boilerplate for c in about), "standing text not flagged"


def test_parent_ids_group_by_section():
    chunks = SitrepChunker(CFG).chunk_document(DOC)
    for c in chunks:
        assert c.parent_id == f"{c.report_id}:s{c.section_index}"


if __name__ == "__main__":
    import json
    chunker = SitrepChunker(CFG)
    cs = chunker.chunk_document(DOC)
    print(f"{len(cs)} chunks from 1 document\n" + "=" * 78)
    for c in cs:
        print(f"\n{c.chunk_id}  sector={c.sector!r:<20} tokens={c.n_tokens:<4} "
              f"table={c.contains_table} figures={c.contains_figures}")
        print("-" * 78)
        print(c.text[:340].rstrip() + ("…" if len(c.text) > 340 else ""))
    print("\n" + "=" * 78)
    print(json.dumps(summarize_stub := {
        "chunks": len(cs),
        "sectors": sorted({c.sector for c in cs if c.sector}),
        "token_range": [min(c.n_tokens for c in cs), max(c.n_tokens for c in cs)],
    }, indent=2))


def test_cli_dispatch_table_resolves():
    """Guards the gap that let cmd_profile silently disappear.

    Nothing else in the suite imports ingest.py, so a function could vanish
    from it and every test would still pass. This is cheap insurance.
    """
    import src.ingest as ing
    for name in ("cmd_sources", "cmd_fetch", "cmd_resolve", "cmd_profile"):
        assert callable(getattr(ing, name, None)), f"{name} missing from ingest.py"


def test_config_loads_with_date_bounds_at_either_level(tmp_path):
    import yaml as _y
    from src.ingest import load_config
    base = _y.safe_load(open("config/corpus.yaml"))
    for where in ("corpus", "filters"):
        cfg = _y.safe_load(open("config/corpus.yaml"))
        c = cfg["corpus"]
        for k in ("date_from", "date_to"):
            c.pop(k, None); c.get("filters", {}).pop(k, None)
        target = c if where == "corpus" else c["filters"]
        target["date_from"] = "2020-01-01"
        pth = tmp_path / f"{where}.yaml"
        pth.write_text(_y.safe_dump(cfg))
        got = load_config(pth)["corpus"]["filters"]
        assert got["date_from"] == "2020-01-01"
        assert got["date_to"]


# --- vocabulary learned from the real corpus -------------------------------
# Every heading below was produced by `ingest profile` over actual extracted
# WHO and UNHCR PDFs. They are regression cases, not invented examples: the
# original OCHA-derived vocabulary matched none of them.

import pytest  # noqa: E402

from src.chunker import (  # noqa: E402
    classify_region, classify_regions, clean_heading, is_furniture,
)


@pytest.mark.parametrize("heading,sector", [
    ("**Laboratory activities**", "Surveillance"),
    ("Surveillance and laboratory investigations", "Surveillance"),
    ("**<mark>Proportionate morbidity</mark>**", "Surveillance"),
    ("Districts Reporting Cases in Week 5", "Surveillance"),
    ("**Suspected Measles Cases**", "Disease Outbreak"),
    # Previously asserted "Disease Outbreak" — which encoded the alternation-
    # precedence bug rather than the right answer. The heading is about case
    # management; cholera is its object. Earliest term now wins.
    ("**Case management and Oral cholera vaccination**", "Case Management"),
    ("New arrivals", "Assistance"),
    ("Cash assistance", "Assistance"),
    ("Community Empowerment and Self-reliance", "Livelihoods"),
    ("**Small-business programme**", "Livelihoods"),
    ("<u>Highlights</u>", "Highlights"),
])
def test_real_headings_classify(heading, sector):
    from src.chunker import classify_sector
    assert classify_sector(heading) == sector


@pytest.mark.parametrize("heading,region", [
    ("**Hiraan region**", "Hiraan"),
    ("**Banadir region** .", "Banadir"),
    ("**Lower Jubba**", "Lower Juba"),
    ("**Middle Shabelle** .", "Middle Shabelle"),
    ("Mogadishu IDP sites", "Banadir"),      # town maps to its admin1
    ("Baidoa reception centre", "Bay"),
    ("Kismayo returnees", "Lower Juba"),
])
def test_regions_detected(heading, region):
    assert classify_region(heading) == region


def test_regions_ordered_by_appearance_not_dict_order():
    got = classify_regions("**AWD/cholera situation in Hiraan and Banadir**")
    assert got == ["Hiraan", "Banadir"]


def test_furniture_headings_flagged():
    for h in ("**CONTACT**", "**LINKS**",
              "**Donors who have contributed to the operation in 2017**",
              "About OCHA", "Annexes"):
        assert is_furniture(h), h
    for h in ("**Laboratory activities**", "Cash assistance", "Health Cluster"):
        assert not is_furniture(h), h


def test_emphasis_stripped_from_stored_headings():
    assert clean_heading("**<mark>Cumulative figures as of week 5</mark>**") \
        == "Cumulative figures as of week 5"



# --- second pass: 527 real documents ---------------------------------------
# Every heading here appeared in the "matching NO sector" list of the second
# profile run, most of them repeatedly (counts in comments).

@pytest.mark.parametrize("heading,sector", [
    ("Completeness of reporting", "Surveillance"),                     # 54
    ("Proportional morbidity", "Surveillance"),                        # 24
    ("Environmental Surveillance", "Surveillance"),                    # 15
    ("cVDPV2", "Disease Outbreak"),                                    # 15
    ("Whooping cough (Perusis) update", "Disease Outbreak"),           # 10
    ("Other Acute Diarrhea Situation in Somalia", "Disease Outbreak"), # 9
    ("Severe Acute respiratory Illness (SARI)", "Disease Outbreak"),   # 9
    ("Influenza like Illness (ILI)", "Disease Outbreak"),              # 9
    ("Public Health Response actions", "Case Management"),             # 7
    ("Upcoming vaccination activities", "Case Management"),            # 6
    ("Communication for Development", "Community Engagement"),         # 8
    ("PEOPLE ASSISTED", "Assistance"),                                 # 10
    ("FIGURES", "Key Figures"),                                        # 6
    ("Financial Information", "Funding"),                              # 13
])
def test_second_pass_headings(heading, sector):
    from src.chunker import classify_sector
    assert classify_sector(heading) == sector


def test_alternation_precedence_regression():
    """The anchor once bound only to the first alternative of each pattern.

    'Environmental Surveillance' failed while 'Environmental laboratory'
    matched — the same rule applied or not depending on list order.
    """
    from src.chunker import classify_sector
    assert classify_sector("Environmental Surveillance") == "Surveillance"
    assert classify_sector("Environmental laboratory") == "Surveillance"


def test_bare_access_is_not_coordination():
    from src.chunker import classify_sector
    assert classify_sector("Access to clean water") is None
    assert classify_sector("Humanitarian access constraints") == "Coordination"


@pytest.mark.parametrize("heading", [
    "Relevant Links", "Tweet of the Week",
    "UNEARMARKED CONTRIBUTIONS | USD", "BROADLY EARMARKED CONTRIBUTIONS | USD",
])
def test_second_pass_furniture(heading):
    """UNHCR donor tables carry figures, so only the furniture rule stops them."""
    assert is_furniture(heading)


# --- third pass: 900 documents ----------------------------------------------

@pytest.mark.parametrize("heading,sector", [
    ("Background", "Situation Overview"),                              # 59
    ("Response", "Response"),                                          # 50
    ("Needs", "Needs"),                                                # 47
    ("Achievements and Impact", "Response"),                           # 42
    ("Gaps & Constraints", "Gaps & Constraints"),                      # 36
    ("_Epidemiological Bulletin_", "Surveillance"),                    # 30
    ("Urgent Needs", "Needs"),                                         # 30
    ("SUMMARY STATISTICS FOR DROUGHTAFFECTED DISTRICTS", "Key Figures"),  # 28
    ("Current situation", "Situation Overview"),                       # 25
    ("Investigation of suspected epidemic notifiable alerts", "Surveillance"),  # 23
    ("Stool Adequacy Rate by District, Somalia, 2022-2024", "Surveillance"),    # 20
    ("Trauma case monitoring and critical care", "Case Management"),   # 15
    ("Top Completed Replies by Type of Feedback", "Accountability"),   # 15
    ("Legal assistance", "Protection"),                                # 14
])
def test_third_pass_headings(heading, sector):
    from src.chunker import classify_sector
    assert classify_sector(heading) == sector


def test_longest_match_wins_a_positional_tie():
    """Specific beats generic when two patterns start at the same place."""
    from src.chunker import classify_sector
    assert classify_sector("Summary Statistics") == "Key Figures"   # not Highlights
    assert classify_sector("Case Response") == "Case Management"    # not Response


@pytest.mark.parametrize("heading", [
    "CONTACTS", "Information Sources", "Table of contents",
    "External / Donors Relations", "@WHO Somalia WHO Somalia somaliawho",
])
def test_third_pass_furniture(heading):
    assert is_furniture(heading)


def test_states_expand_to_constituent_regions():
    assert classify_regions("Hirshabelle State") == ["Hiraan", "Middle Shabelle"]
    assert classify_regions("Galmudug State") == ["Galgaduud", "Mudug"]


def test_contested_regions_are_not_assigned_to_either_state():
    """Sool and Sanaag are contested; the facet must not take a side."""
    claimed = classify_regions("Puntland") + classify_regions("Somaliland")
    assert "Sool" not in claimed and "Sanaag" not in claimed


def test_numeric_heading_does_not_orphan_the_figure():
    """'# 2163' is a key figure in a big font, not a section boundary."""
    secs = split_sections("## Key figures\n\n# 2163\n\nsuspected cholera cases\n\n"
                          "## Response\n\nPartners treated patients.")
    assert [x.heading for x in secs] == ["Key figures", "Response"]
    assert "2163" in secs[0].text and "suspected cholera" in secs[0].text


# --- resolution: structure beats length --------------------------------------

EX = {"body_min_chars": 2500, "merge_body_and_pdf": False,
      "prefer_structured": True, "min_body_sections": 3}


def test_long_unstructured_body_goes_to_pdf():
    """The OCHA case: long, but a prose rendering with the headings flattened."""
    from src.ingest import _needs_pdf
    prose = ("Humanitarian partners continued to scale up the response. " * 60)
    assert len(prose) > 2500
    assert _needs_pdf({"body": prose}, EX)


def test_long_structured_body_is_kept():
    from src.ingest import _needs_pdf
    body = "\n\n".join(f"## {h}\n\n" + "Partners reported progress. " * 40
                       for h in ("Highlights", "Health", "Nutrition", "Protection"))
    assert len(body) > 2500
    assert not _needs_pdf({"body": body}, EX)


def test_thin_body_always_goes_to_pdf():
    from src.ingest import _needs_pdf
    assert _needs_pdf({"body": "A short teaser."}, EX)


def test_truncated_pdf_is_detected(tmp_path):
    """An interrupted download keeps the %PDF header and loses the %%EOF trailer.

    Before this check, a truncated file looked cached, was never re-fetched,
    and its extraction failed silently on every subsequent run.
    """
    import pymupdf
    from src.ingest import _pdf_looks_complete

    d = pymupdf.open(); d.new_page().insert_text((50, 60), "sitrep")
    good = d.tobytes()
    ok, cut = tmp_path / "ok.pdf", tmp_path / "cut.pdf"
    ok.write_bytes(good)
    cut.write_bytes(good[: len(good) // 2])
    assert _pdf_looks_complete(ok)
    assert not _pdf_looks_complete(cut)


# --- CLI entry points --------------------------------------------------------

def test_every_cli_entry_point_runs():
    """Guards forward-reference bugs that only bite when run as __main__.

    `section_is_furniture` was defined BELOW the `if __name__ == "__main__"`
    block. Importing the module defined it fine, which is how the scale test
    ran; `python -m src.chunker` executed the main block first and raised
    NameError. Import-based tests cannot see this class of bug, so these run
    the modules the way a user does.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for module in ("src.ingest", "src.chunker", "src.embed", "src.search"):
        r = subprocess.run([sys.executable, "-m", module, "--help"],
                           cwd=root, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{module} --help failed:\n{r.stderr[-1500:]}"


def test_chunker_cli_runs_end_to_end(tmp_path):
    """The full main() path: config -> documents.jsonl -> chunks.jsonl."""
    import json
    import subprocess
    import sys
    import yaml as _y
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = _y.safe_load((root / "config" / "corpus.yaml").read_text())
    for key in ("raw", "interim", "processed", "pdf_cache"):
        cfg["paths"][key] = str(tmp_path / key)
    (tmp_path / "interim").mkdir(parents=True)
    (tmp_path / "interim" / "documents.jsonl").write_text(json.dumps({
        **DOC, "text_source": "pdf"}) + "\n")
    cfg_path = tmp_path / "corpus.yaml"
    cfg_path.write_text(_y.safe_dump(cfg))

    r = subprocess.run([sys.executable, "-m", "src.chunker", "--config", str(cfg_path)],
                       cwd=root, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    chunks = [json.loads(l) for l in
              (tmp_path / "processed" / "chunks.jsonl").read_text().splitlines() if l]
    assert chunks
    assert all(c["date_original"][:10] in c["text"] for c in chunks)


def test_oversized_table_splits_at_rows_keeping_the_header():
    """A table fragment without its header is numbers with no column labels.

    Emitting oversized tables whole was worse than splitting them: the
    embedding model truncates at its window with no error, so the tail was
    never embedded while still appearing in the text.
    """
    rows = "\n".join(f"| District {i} | {1000 + i * 37} | {i % 9}.{i % 7} per cent |"
                     for i in range(120))
    table = "| District | Cases | CFR |\n| --- | --- | --- |\n" + rows
    out = SitrepChunker(CFG).chunk_document({**DOC, "text": "## Surveillance\n\n" + table})
    assert len(out) > 1
    assert all("| District | Cases | CFR |" in c.body for c in out)
    assert all(c.n_tokens <= CFG["chunking"]["max_tokens"] for c in out)
    assert sum(c.body.count("\n") - 1 for c in out) == 120   # no rows lost


def test_token_counts_are_not_assumed_additive():
    """Sizing by summing per-part counts under-estimates the assembled text."""
    ch = SitrepChunker(CFG)
    parts = [f"| District {i} | {1000 + i} | 1.2 per cent |" for i in range(40)]
    assert ch.count("\n".join(parts)) >= sum(ch.count(p) for p in parts) - 5


def test_run_on_sentence_without_terminator_is_still_split():
    """Extracted PDF text can run hundreds of tokens with no full stop."""
    ch = SitrepChunker(CFG)
    runon = " ".join(f"Banadir district {i} reported {100 + i} cases" for i in range(300))
    out = ch.chunk_document({**DOC, "text": "## Surveillance\n\n" + runon})
    assert all(c.n_tokens <= CFG["chunking"]["max_tokens"] for c in out)

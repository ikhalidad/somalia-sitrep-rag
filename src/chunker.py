"""
Stage 3: structure-aware chunking for humanitarian situation reports.

Why not RecursiveCharacterTextSplitter
--------------------------------------
Situation reports are not prose. They are a fixed skeleton — HIGHLIGHTS,
KEY FIGURES, SITUATION OVERVIEW, then one block per cluster (Health, WASH,
Nutrition, Protection, Food Security...), then FUNDING — refilled every
fortnight with new numbers. Two consequences drive every decision here.

1. SECTION BOUNDARIES ARE SEMANTIC BOUNDARIES.
   A fixed-size splitter puts the tail of the Nutrition update and the head
   of the Protection update in one chunk. Retrieval then returns a chunk
   that reads as if a GBV caseload figure belongs to a nutrition programme.
   That is not a subtle quality loss; it is a wrong answer with a citation
   attached. So: split on sections first, size second, never across.

2. THE TEMPORAL COLLISION PROBLEM — the defining failure mode of this corpus.
   "An estimated 3.8 million people remain internally displaced" appears in
   every edition with a different figure. Those chunks are near-identical in
   embedding space (cosine > 0.97). Ask "how many IDPs are there?" and the
   retriever returns a semantically arbitrary edition, possibly from 2019.
   The LLM then answers faithfully and cites correctly — and is wrong by
   four years.

   Note what this defeats: Ragas `faithfulness` scores this answer 1.0,
   because the answer IS supported by the retrieved context. The bug lives
   in retrieval, not generation, and only context_precision against a
   date-aware ground truth catches it.

   Three mitigations, all applied at chunk time because none can be
   retrofitted at query time:
     a) the publication date is written INTO the embedded text, so the
        vector itself carries temporal signal;
     b) date, source and section are emitted as filterable metadata, so the
        retriever can hard-filter a "March 2025" question to March 2025;
     c) every chunk is self-dating, so the generator can be required to
        state an as-of date and will visibly fail rather than silently
        mislead when it cannot.

Small-to-big
------------
Embed the ~400-token child, hand the parent section to the generator. The
child is specific enough to match a question; the parent is complete enough
to answer it. `parent_id` on every chunk supports this at retrieval time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger("chunker")

# --------------------------------------------------------------------------
# section vocabulary
# --------------------------------------------------------------------------

# Cluster / sector names as they actually appear in Somalia sitreps across
# OCHA, UNHCR, WHO, UNICEF, WFP and IOM DTM products. Used both to detect
# unmarked sections and to tag chunks with a normalised `sector` facet,
# which is the filter a humanitarian user reaches for first.
SECTOR_PATTERNS: dict[str, str] = {
    "Health": r"health(?:\s+cluster|\s+sector)?",
    "Nutrition": r"nutrition(?:\s+cluster|\s+sector)?",
    "WASH": r"wash|water,?\s*sanitation(?:\s*(?:and|&)\s*hygiene)?",
    "Protection": r"protection(?:\s+cluster|\s+sector)?|\bprotect\b|gbv|gender[- ]based\s+violence"
                  r"|child\s+protection|legal\s+(?:assistance|aid|counsel+ing)"
                  r"|civil\s+documentation|housing,?\s+land",
    "Food Security": r"food\s+security(?:\s+(?:and|&)\s+livelihoods)?|fsl",
    "Shelter/NFI": r"shelter(?:\s*/?\s*nfi)?|non[- ]food\s+items",
    "Education": r"education(?:\s+cluster|\s+sector)?",
    "CCCM": r"cccm|camp\s+coordination",
    "Logistics": r"logistics(?:\s+cluster)?",
    "Displacement": r"displacement|idp|internally\s+displaced|returnee|prmn",
    "Funding": r"funding|financial\s+(?:requirements|overview|information)"
               r"|appeal|hrp|financing",
    "Highlights": r"highlights?|key\s+messages?|at\s+a\s+glance|summary",
    "Key Figures": r"key\s+figures?|by\s+the\s+numbers|in\s+numbers"
                   r"|cumulative\s+figures|^figures$|summary\s+(?:of\s+)?statistics"
                   r"|population\s+(?:data|figures)|indicators?|demographics?",
    # "Situation Updates" (10) missed the whole corpus over a missing "s?".
    "Situation Overview": r"situation\s+(?:overviews?|updates?|analysis)"
                          r"|current\s+situation|general\s+situation"
                          r"|updates?\s+by\s+(?:area|region)|background|context|overview",
    # Bare "access" matched "Access to clean water" as Coordination.
    "Coordination": r"coordination|humanitarian\s+access|access\s+constraints?"
                    r"|access\s+(?:and|&)\s+security|access\s+impediments?"
                    r"|civil[- ]military",

    # --- learned from the corpus, not assumed -------------------------------
    # WHO is 49% of this corpus and publishes EWARN epidemiological bulletins,
    # not OCHA-style cluster sitreps. These headings came out of the profile
    # run over real extracted PDFs; none of them would have matched the
    # OCHA-derived vocabulary above.
    # Second pass, from 527 documents. "Proportional" not "proportionate" is
    # what WHO actually writes; completeness-of-reporting is the single most
    # common unmatched heading in the corpus (54 occurrences).
    "Surveillance": r"surveillance|laboratory|lab\s+activities"
                    r"|proportion(?:al|ate)\s+morbidity|districts?\s+reporting"
                    r"|completeness\s+of\s+reporting|reporting\s+rate"
                    r"|epidemiologic(?:al)?\s+(?:updates?|week|bulletin|situation"
                    r"|curve|monitor)"
                    r"|notifiable\s+alerts?|investigation\s+of\s+suspected"
                    r"|stool\s+adequacy|afp|acute\s+flaccid\s+paralysis"
                    r"|alerts?\s+(?:and\s+)?(?:response|verification)",
    "Disease Outbreak": r"awd|(?:acute\s+)?(?:watery\s+)?diarrho?ea|cholera|measles"
                        r"|malaria|polio|poliovirus|cvdpv\d?|vaccine[- ]derived"
                        r"|diphtheria|covid|dengue|whooping\s+cough|pertussis|perusis"
                        r"|influenza|ili|sari|severe\s+acute\s+respiratory"
                        r"|respiratory\s+illness|outbreak|suspected\s+cases"
                        r"|case\s+fatality|attack\s+rate",
    "Case Management": r"case\s+management|case\s+response|public\s+health\s+response"
                       r"|response\s+actions|oral\s+cholera\s+vaccination|ocv"
                       r"|treatment\s+cent(?:re|er)|vaccination|immuni[sz]ation"
                       r"|trauma|critical\s+care|mass\s+casualty",
    "Community Engagement": r"communication\s+for\s+development|c4d|risk\s+communication"
                            r"|community\s+engagement|rcce|social\s+mobili[sz]ation",
    # Generic humanitarian report sections, the commonest unmatched headings
    # in the third pass: Response (50), Needs (47), Achievements (42),
    # Gaps & Constraints (36), Urgent Needs (30).
    # UNHCR structures operational updates around four strategic directions,
    # rendered as "1. PROTECT" / "2. RESPOND" / "3. EMPOWER" / "4. SOLVE"
    # (15 occurrences each).
    "Response": r"response|respond|achievements?(?:\s+and\s+impact)?|interventions?"
                r"|updates?\s+on\s+achievements?|capacity\s+building",
    "Durable Solutions": r"durable\s+solutions?|\bsolve\b|reintegration"
                         r"|resettlement|voluntary\s+repatriation|returns?\s+programme",
    "Needs": r"(?:urgent\s+|priority\s+|humanitarian\s+)?needs",
    "Gaps & Constraints": r"gaps?(?:\s*(?:&|and)\s*constraints?)?|constraints?"
                          r"|challenges?|bottlenecks?",
    # CCCM and UNHCR feedback dashboards — accountability to affected people.
    "Accountability": r"feedback|complaints?|accountability|aap|hotline",
    # UNHCR is 39% and publishes operational updates structured by programme.
    "Assistance": r"cash\s+assistance|new\s+arrivals?|people\s+assisted|returnees?\s+assist"
                  r"|core\s+relief|distribution|registration|reception\s+cent",
    "Livelihoods": r"livelihoods?|self[- ]reliance|empower(?:ment)?"
                   r"|vocational|small[- ]business|income\s+generat",
}

# Headings that are document furniture rather than content. Matching one makes
# a section a drop candidate regardless of length — UNHCR operational updates
# end every edition with the same donor list and contact block.
FURNITURE_PATTERNS = re.compile(
    r"^\W*(?:contacts?|(?:relevant\s+|useful\s+|key\s+)?links?|donors?\b"
    r"|information\s+sources?|sources?\s+of\s+information|data\s+sources?"
    r"|references?|table\s+of\s+contents?|contents|in\s+this\s+issue"
    r"|editorial\s+note|monthly\s+reports?|dr\.?\s+\w+\s+\w+"
    r"|external\s*(?:/|and|&)?\s*donors?(?:\s+relations?)?"
    r"|acknowledge?ments?|about\s+(?:us|ocha|unhcr|who)|tweet\s+of\s+the\s+week"
    # UNHCR's global donor tables. They carry figures, so the boilerplate
    # rule's "has figures => substantive" guard would let them through — but
    # they list UNHCR's worldwide funding, identical in every country update,
    # and say nothing about Somalia.
    r"|(?:un|broadly\s+|softly\s+|tightly\s+)?earmarked\s+contributions?"
    r"|for\s+(?:more|further)\s+information|media\s+contact|follow\s+us|disclaimer"
    r"|annex(?:es)?\b|abbreviations?|acronyms?)\b", re.I)

# --- Somalia geography -----------------------------------------------------
# Regions appear as section headings in WHO bulletins ("**Hiraan region**",
# "**Banadir region**") and are the single most common qualifier in questions
# about this corpus — "cholera in Banadir", "displacement in Gedo". Making
# region a retrieval facet is worth more here than any amount of reranking.
# Major towns map to their admin1 because reports use them interchangeably.
REGION_PATTERNS: dict[str, str] = {
    "Awdal": r"awdal|borama",
    "Bakool": r"bakool|xudur|hudur|waajid",
    "Banadir": r"banadir|benadir|mogadishu|muqdisho",
    "Bari": r"\bbari\b|bosaso|bossaso|qardho",
    "Bay": r"\bbay\b|baidoa|baydhabo|burhakaba",
    "Galgaduud": r"galgaduud|galgadud|dhusamareb|dhuusamarreeb|cadaado|abudwak",
    "Gedo": r"gedo|dolow|doolow|garbaharey|luuq|belet\s?hawo",
    "Hiraan": r"hiraan|hiran|beled\s?weyne|belet\s?weyne|bulo\s?burte",
    "Lower Juba": r"lower\s+(?:juba|jubba)|jubbada\s+hoose|kismayo|kismaayo|afmadow",
    "Middle Juba": r"middle\s+(?:juba|jubba)|jubbada\s+dhexe|bu.?aale|jilib|sakow",
    "Lower Shabelle": r"lower\s+shabe+lle?|shabeellaha\s+hoose|marka|merca|afgooye|qoryoley",
    "Middle Shabelle": r"middle\s+shabe+lle?|shabeellaha\s+dhexe|jowhar|balcad|adale",
    "Mudug": r"mudug|galkacyo|gaalkacyo|galkayo|hobyo|harardhere",
    "Nugaal": r"nugaal|nugal|garowe|garoowe|burtinle|eyl",
    "Sanaag": r"sanaag|erigavo|ceerigaabo|badhan|laasqoray",
    "Sool": r"\bsool\b|las\s?anod|laascaanood|taleex|xudun",
    "Togdheer": r"togdheer|burco|burao|oodweyne|buhoodle",
    "Woqooyi Galbeed": r"woqooyi\s+galbeed|hargeisa|hargeysa|berbera|gabiley",
}
_REGION_RE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in REGION_PATTERNS.items()}

# Reports organise by Federal Member State as often as by region ("Galmudug
# State" 16, "Hirshabelle State" 13 in the third pass). A question about
# Hirshabelle should reach chunks about Hiraan and Middle Shabelle, so a state
# expands to its constituent regions.
#
# Sool and Sanaag are deliberately absent from both Puntland and Somaliland.
# Their administration is contested, and encoding either claim in a retrieval
# facet is a political position this system has no business taking. Mudug is
# listed under both Galmudug and Puntland because it is divided between them.
STATE_REGIONS: dict[str, list[str]] = {
    r"galmudug": ["Galgaduud", "Mudug"],
    r"hirshabelle|hir[- ]?shabelle": ["Hiraan", "Middle Shabelle"],
    r"south[\s-]*west\s+state|southwest\s+state": ["Bay", "Bakool", "Lower Shabelle"],
    r"jubb?aland": ["Gedo", "Lower Juba", "Middle Juba"],
    r"puntland": ["Bari", "Nugaal", "Mudug"],
    r"somaliland": ["Awdal", "Woqooyi Galbeed", "Togdheer"],
}
_STATE_RE = [(re.compile(rf"\b(?:{k})\b", re.I), v) for k, v in STATE_REGIONS.items()]
# Every pattern is wrapped in a non-capturing group. The original form,
# rf"^\W*{v}\b", looked anchored but was not: alternation binds loosest, so
# `^` attached to the FIRST alternative only. "Environmental Surveillance"
# failed while "Environmental laboratory" matched — same rule, applied or not
# depending on which word happened to be listed first.
_SECTOR_RE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in SECTOR_PATTERNS.items()}

# Unmarked headings: ALL-CAPS or Title Case lines, short, no terminal period.
# PDF extraction loses some `#` markers, so this is the safety net.
_BARE_HEADING = re.compile(r"^\s{0,3}([A-Z][A-Za-z/&'’\- ]{2,70})\s*:?\s*$")
_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_NUMERIC_HEADING = re.compile(r"^[\d\s,.%$+\-–]+$")
_BULLET = re.compile(r"^\s*([-*•‣]|\d+[.)])\s+")

# Strip the furniture that repeats in every edition and matches nothing useful.
_NOISE = [
    (re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.I | re.M), ""),
    (re.compile(r"^\s*-{3,}\s*$", re.M), ""),
    (re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$", re.M), ""),          # images
    (re.compile(r"\n{4,}"), "\n"),
]

# PDF extraction artefacts. pymupdf wraps image captions in HTML comments and
# renders line breaks and highlights as tags: "<!-- Start of picture text -->",
# "<br>", "<mark>". They are noise in the embedded vector and, worse, they are
# noise in a quoted citation, which is what a reader actually sees.
_ARTEFACTS = [
    (re.compile(r"<!--.*?-->", re.S), " "),
    (re.compile(r"</?(?:br|mark|u|b|i|em|strong|span|sup|sub|p|div)[^>]*>", re.I), " "),
    (re.compile(r"[ \t]{2,}"), " "),
]


def tidy(text: str) -> str:
    """Remove extraction artefacts from text about to be shown or quoted.

    Applied at display and prompt time as well as at chunk time, so an index
    built before this existed still produces clean citations without a
    hundred-minute re-embed.
    """
    for pat, repl in _ARTEFACTS:
        text = pat.sub(repl, text)
    return text.strip()


# --------------------------------------------------------------------------
# token counting
# --------------------------------------------------------------------------

def get_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Return a token counter for the EMBEDDING model, not a generic one.

    This matters more than it looks. bge-small-en-v1.5 accepts 512 tokens
    and silently truncates beyond that — no exception, no warning, just a
    vector built from two thirds of your chunk. Sizing chunks with a
    word-count heuristic while embedding with a 512-token model is the
    single most common invisible bug in RAG pipelines.
    """
    if model_name:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(model_name)
            # Counting, not inferring: sequences longer than the model window
            # are exactly what we are here to detect and split, so the
            # "longer than maximum sequence length" warning is noise.
            tok.model_max_length = int(1e9)
            return lambda s: len(tok.encode(s, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001
            log.warning("tokenizer %s unavailable (%s) — heuristic fallback",
                        model_name, type(exc).__name__)

    def approx(s: str) -> int:
        # Deliberately over-estimates. Sitreps are dense with numerals,
        # acronyms and place names, all of which fragment into several
        # word-pieces. Under-estimating means silent truncation.
        words = s.split()
        return int(len(words) * 1.45) + sum(c.isdigit() for c in s) // 3

    return approx


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass
class Section:
    heading: str
    sector: str | None
    level: int
    text: str
    index: int
    regions: list[str] = field(default_factory=list)
    furniture: bool = False


@dataclass
class Chunk:
    chunk_id: str
    report_id: int
    parent_id: str
    text: str            # what gets embedded: header + body
    body: str            # body only, for display
    header: str
    # -- retrieval facets --
    date_original: str | None
    year_month: str | None
    sources: list[str]
    sector: str | None
    regions: list[str]
    heading: str | None
    title: str | None
    url: str | None
    text_source: str | None
    is_primary_country_som: bool
    # -- bookkeeping --
    section_index: int
    chunk_index: int
    n_tokens: int
    contains_table: bool
    contains_figures: bool
    content_sha256: str
    content_group: str | None = None   # shared by identical text across editions
    n_editions: int = 1                # how many documents carry this exact text
    is_boilerplate: bool = False
    duplicate_of: str | None = None
    flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    for pat, repl in _NOISE:
        text = pat.sub(repl, text)
    return tidy(text)


_EMPHASIS_RE = re.compile(r"\*\*|__|</?(?:u|b|i|em|strong|mark|span)[^>]*>|`")


def clean_heading(heading: str) -> str:
    """Strip the markup PDF extraction leaves around headings.

    pymupdf4llm renders a bold heading as `#### **Laboratory activities**`,
    and underline or highlight as `<u>Highlights</u>` / `<mark>...</mark>`.
    Left in place the markers reach the stored heading, the citation header
    and the embedded text — three places where `**` is noise.
    """
    # Single "_" italics are stripped too: "_Epidemiological Bulletin_" beat
    # every pattern, because "_" is a word character and \b never fires
    # between it and the "E".
    return _EMPHASIS_RE.sub("", heading).strip(" .:*#-—_").strip()


def classify_sector(heading: str) -> str | None:
    """The sector whose term appears EARLIEST in the heading.

    Real headings name several things: "Case management and Oral cholera
    vaccination" is about case management; cholera is the object. First
    match in dictionary order would make the answer depend on how the
    pattern table happens to be sorted. Earliest position in the heading is
    what a reader uses, and it is stable under reordering.
    """
    h = clean_heading(heading)
    best: tuple[int, int, str] | None = None
    for sector, rx in _SECTOR_RE.items():
        m = rx.search(h)
        if not m:
            continue
        # Earliest position first; at the same position the LONGER match wins,
        # so "Summary Statistics" is Key Figures rather than Highlights'
        # bare "summary", and "Case response" is Case Management rather than
        # the generic Response.
        key = (m.start(), -(m.end() - m.start()), sector)
        if best is None or key[:2] < best[:2]:
            best = key
    return best[2] if best else None


def classify_regions(text: str, limit: int = 6) -> list[str]:
    """Somalia admin1 regions named in the text, in order of first appearance.

    Ordered by position rather than by dictionary order, so a heading reading
    "AWD/cholera situation in Hiraan and Banadir" leads with Hiraan. Returns a
    list because a section often covers several regions and a single-value
    facet would silently discard the rest.
    """
    found: list[tuple[int, int, str]] = []
    for region, rx in _REGION_RE.items():
        m = rx.search(text)
        if m:
            found.append((m.start(), 0, region))
    for rx, regions in _STATE_RE:
        m = rx.search(text)
        if m:
            found.extend((m.start(), i + 1, r) for i, r in enumerate(regions))
    out: list[str] = []
    for _, _, r in sorted(found):
        if r not in out:
            out.append(r)
    return out[:limit]


def classify_region(text: str) -> str | None:
    regions = classify_regions(text)
    return regions[0] if regions else None


_HANDLE_RE = re.compile(r"@\w{2,}")


def is_furniture(heading: str) -> bool:
    """Contact blocks, link lists, donor tables, social handles."""
    h = clean_heading(heading)
    # "@WHO Somalia WHO Somalia somaliawho" — a social media footer that
    # pymupdf promotes to a heading because it is set large.
    return bool(FURNITURE_PATTERNS.match(h) or _HANDLE_RE.search(h))


def split_sections(text: str) -> list[Section]:
    """Split a document into sections on Markdown headings, then bare headings."""
    lines = clean_text(text).split("\n")
    sections: list[Section] = []
    cur_head, cur_level, buf = "Preamble", 0, []
    in_table = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if not body:
            return
        head = clean_heading(cur_head) or cur_head
        # Region from the heading first; the body only as fallback. A heading
        # saying "Banadir region" is about Banadir. A body mentioning Banadir
        # once in passing usually is not.
        regions = classify_regions(head) or classify_regions(body[:600])
        sections.append(Section(head, classify_sector(head), cur_level, body,
                                len(sections), regions, is_furniture(head)))

    for line in lines:
        if _TABLE_ROW.match(line):
            in_table = True
            buf.append(line)
            continue
        if in_table and not _TABLE_ROW.match(line):
            in_table = False

        m = _MD_HEADING.match(line)
        if m and _NUMERIC_HEADING.match(clean_heading(m.group(2))):
            # "# 2163" — a key figure set in a large font, which pymupdf
            # promotes to a heading. Splitting on it would orphan the number
            # from the label that gives it meaning.
            buf.append(clean_heading(m.group(2)))
            continue
        if m:
            flush()
            cur_level, cur_head, buf = len(m.group(1)), m.group(2).strip(), []
            continue

        # A bare heading only counts if it names a known section. Otherwise
        # every short capitalised line — place names, org names — shatters
        # the document into unusable fragments.
        bare = _BARE_HEADING.match(line)
        if bare and not _BULLET.match(line) and classify_sector(bare.group(1)):
            flush()
            cur_level, cur_head, buf = 2, bare.group(1).strip(), []
            continue

        buf.append(line)
    flush()
    return sections or [Section("Document", None, 0, clean_text(text), 0)]


def split_blocks(text: str) -> list[tuple[str, bool]]:
    """Section body → (block, is_table) units. Tables stay whole."""
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    table: list[str] = []

    def flush_text() -> None:
        if buf and "\n".join(buf).strip():
            blocks.append(("\n".join(buf).strip(), False))
        buf.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("\n".join(table).strip(), True))
        table.clear()

    for line in text.split("\n"):
        if _TABLE_ROW.match(line):
            flush_text()
            table.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_text()
        else:
            buf.append(line)
    flush_text()
    flush_table()
    return blocks


# --------------------------------------------------------------------------
# the contextual header
# --------------------------------------------------------------------------

def build_header(doc: dict[str, Any], section: Section) -> str:
    """Prefix prepended to every chunk BEFORE embedding.

    Deterministic, not LLM-generated: it costs nothing, never hallucinates,
    and is reproducible across re-indexes. It does three jobs at once —
    carries the date into the vector (mitigation (a) above), disambiguates
    otherwise-identical chunks from different editions, and makes the
    retrieved context self-describing so the generator can cite it without
    a separate metadata lookup.
    """
    bits = []
    d = (doc.get("date_original") or "")[:10]
    if d:
        bits.append(d)
    src = "/".join(filter(None, (doc.get("source_shortnames")
                                 or doc.get("sources") or [])[:2]))
    if src:
        bits.append(src)
    bits.append("Somalia Situation Report")
    if section.sector:
        bits.append(section.sector)
    elif section.heading and section.heading != "Preamble":
        bits.append(section.heading[:60])
    if section.regions:
        bits.append("/".join(section.regions[:2]))
    title = (doc.get("title") or "").strip()
    head = " | ".join(bits)
    return f"[{head}]\n{title}\n" if title else f"[{head}]\n"


# Figure detection, tuned to the register this corpus is actually written in.
# UN house style spells out "per cent" rather than "%", and reaches for
# "billion" as readily as "million". A detector built on "%" and "million"
# alone reads a funding paragraph — "41 per cent funded against requirements
# of 1.6 billion United States dollars" — as carrying no figures at all, and
# the boilerplate rule then deletes it. Comma-grouped numerals (1,204 /
# 88,000) are included because they are the most reliable figure signal in
# the corpus and need no unit beside them.
_FIGURE_RE = re.compile(
    r"\b\d[\d,.]*\s*(?:per\s?cent|percent|%|billion|million|thousand|m\b|k\b"
    r"|people|persons|individuals|households|cases|children|women|girls|boys"
    r"|idps?|returnees?|districts?|facilities|sites|usd|dollars?)"
    r"|\$\s?\d"
    r"|\b\d{1,3}(?:,\d{3})+\b",
    re.I,
)


# --------------------------------------------------------------------------
# chunker
# --------------------------------------------------------------------------

class SitrepChunker:
    def __init__(self, cfg: dict[str, Any]) -> None:
        c = cfg["chunking"]
        self.count = get_token_counter(c.get("embedding_model"))
        self.max_tokens = c["max_tokens"]
        self.target = c["target_tokens"]
        self.overlap = c["overlap_tokens"]
        self.min_tokens = c["min_tokens"]
        self.atomic_tables = c["atomic_tables"]
        self.dedup_cfg = cfg.get("dedup", {})

    # -- per document ------------------------------------------------------

    def chunk_document(self, doc: dict[str, Any]) -> list[Chunk]:
        text = doc.get("text") or ""
        if not text.strip():
            return []
        out: list[Chunk] = []
        for section in split_sections(text):
            header = build_header(doc, section)
            budget = self.max_tokens - self.count(header)
            if budget < self.min_tokens:
                header, budget = f"[{(doc.get('date_original') or '')[:10]}]\n", \
                                 self.max_tokens - 12
            # Same non-additivity applies when packing prose blocks, so the
            # budget keeps a small margin rather than trusting the running sum.
            budget -= max(8, self.max_tokens // 32)
            parent_id = f"{doc['report_id']}:s{section.index}"
            for i, (body, has_table) in enumerate(
                self._pack(section, min(self.target, budget), budget)
            ):
                out.append(self._make(doc, section, header, parent_id, i, body, has_table))
        return out

    def _pack(self, section: Section, target: int, hard: int
              ) -> Iterator[tuple[str, bool]]:
        """Pack blocks to `target`, never exceeding `hard`, tables atomic."""
        buf: list[str] = []
        buf_tokens = 0
        buf_table = False

        for block, is_table in split_blocks(section.text):
            btoks = self.count(block)

            if is_table and self.atomic_tables:
                if buf:
                    yield "\n\n".join(buf), buf_table
                    buf, buf_tokens, buf_table = [], 0, False
                if btoks <= hard:
                    yield block, True
                else:
                    # Emitting it whole was worse than splitting it. The
                    # embedding model truncates at its window without error,
                    # so an oversized chunk is not "kept intact" — its tail is
                    # simply never embedded, and the rows it contains become
                    # unretrievable while still appearing in the text.
                    for piece in self._split_table(block, hard):
                        yield piece, True
                continue

            if btoks > hard:
                for piece in self._split_oversized(block, target):
                    if buf and buf_tokens + self.count(piece) > hard:
                        yield "\n\n".join(buf), buf_table
                        buf, buf_tokens, buf_table = [], 0, False
                    buf.append(piece)
                    buf_tokens += self.count(piece)
                continue

            if buf_tokens + btoks > target and buf:
                yield "\n\n".join(buf), buf_table
                tail = self._tail(buf, self.overlap)
                buf = list(tail)
                buf_tokens = sum(self.count(t) for t in tail)
                buf_table = False

            buf.append(block)
            buf_tokens += btoks
            buf_table = buf_table or is_table

        if buf and buf_tokens >= self.min_tokens:
            yield "\n\n".join(buf), buf_table
        elif buf:
            yield "\n\n".join(buf), buf_table  # keep short tails; flagged later

    def _split_table(self, block: str, budget: int) -> list[str]:
        """Split a long Markdown table at row boundaries, repeating the header.

        A table fragment without its header row is a grid of numbers with no
        column labels: readable, quotable, and wrong. Repeating the header in
        every piece keeps each row attributable to what it measures.
        """
        lines = [ln for ln in block.split("\n") if ln.strip()]
        sep = (len(lines) > 1
               and set(lines[1].replace("|", "").strip()) <= set("-: "))
        header = lines[:2] if sep else lines[:1]
        head_txt = "\n".join(header)
        head_toks = self.count(head_txt)
        rows = lines[len(header):]

        # Token counts are NOT additive: the assembled piece tokenizes to more
        # than the sum of its rows, because subword merges and whitespace do
        # not partition cleanly. Summing per-row counts to size a piece
        # therefore under-estimates, and pieces come out over budget — which
        # is exactly the silent truncation this method exists to prevent. So
        # build greedily on the estimate, then measure the real thing and give
        # rows back until it fits.
        out: list[str] = []
        i = 0
        while i < len(rows):
            cur: list[str] = []
            cur_toks = 0
            while i < len(rows):
                rt = self.count(rows[i])
                if cur and head_toks + cur_toks + rt > budget:
                    break
                cur.append(rows[i])
                cur_toks += rt
                i += 1
            piece = head_txt + "\n" + "\n".join(cur)
            while len(cur) > 1 and self.count(piece) > budget:
                cur.pop()
                i -= 1
                piece = head_txt + "\n" + "\n".join(cur)
            out.append(piece)
        return out or [block]

    def _hard_wrap(self, text: str, budget: int) -> list[str]:
        """Last resort: split on word boundaries.

        Extracted PDF text sometimes runs hundreds of tokens with no sentence
        terminator at all — a caption, a legend, a run-on list. Sentence
        splitting cannot help there, and the alternative is silent truncation.
        """
        words = text.split()
        out, cur = [], []
        for w in words:
            cur.append(w)
            if self.count(" ".join(cur)) > budget:
                cur.pop()
                if cur:
                    out.append(" ".join(cur))
                cur = [w]
        if cur:
            out.append(" ".join(cur))
        return out or [text]

    def _split_oversized(self, block: str, target: int) -> list[str]:
        """Sentence-boundary split for a single over-long paragraph."""
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block)
        pieces, buf, toks = [], [], 0
        for s in sentences:
            st = self.count(s)
            if toks + st > target and buf:
                pieces.append(" ".join(buf))
                buf, toks = [], 0
            buf.append(s)
            toks += st
        if buf:
            pieces.append(" ".join(buf))
        # A "sentence" can itself exceed the budget; fall through to words.
        final: list[str] = []
        for piece in pieces:
            if self.count(piece) > target:
                final.extend(self._hard_wrap(piece, target))
            else:
                final.append(piece)
        return final

    def _tail(self, blocks: list[str], overlap: int) -> list[str]:
        """Trailing blocks worth `overlap` tokens — whole blocks only.

        Character-window overlap cuts mid-sentence and mid-number; a chunk
        starting "...4 million people" is worse than no overlap at all.
        """
        tail, total = [], 0
        for b in reversed(blocks):
            t = self.count(b)
            if total + t > overlap and tail:
                break
            tail.insert(0, b)
            total += t
        return tail

    def _make(self, doc, section, header, parent_id, idx, body, has_table) -> Chunk:
        full = header + body
        norm = re.sub(r"\s+", " ", body.lower()).strip()
        n_tok = self.count(full)
        flags = []
        if section.furniture:
            flags.append("furniture")
        if n_tok > self.max_tokens:
            flags.append("over_budget")
        if self.count(body) < self.min_tokens:
            flags.append("short")
        date_o = doc.get("date_original")
        return Chunk(
            chunk_id=f"{parent_id}:c{idx}",
            report_id=doc["report_id"],
            parent_id=parent_id,
            text=full,
            body=body,
            header=header.strip(),
            date_original=date_o,
            year_month=date_o[:7] if date_o else None,
            sources=doc.get("source_shortnames") or doc.get("sources") or [],
            sector=section.sector,
            regions=section.regions,
            heading=section.heading,
            title=doc.get("title"),
            url=doc.get("url"),
            text_source=doc.get("text_source"),
            is_primary_country_som=bool(doc.get("is_primary_country_som")),
            section_index=section.index,
            chunk_index=idx,
            n_tokens=n_tok,
            contains_table=has_table,
            contains_figures=bool(_FIGURE_RE.search(body)),
            content_sha256=hashlib.sha256(norm.encode()).hexdigest(),
            flags=flags,
        )

    # -- corpus level ------------------------------------------------------

    def mark_duplicates(self, chunks: list[Chunk]) -> list[Chunk]:
        """Separate true boilerplate from merely-unchanged content.

        The naive rule — drop anything whose text appears in several
        documents — destroys a serial corpus. When OCHA's Health section is
        word-for-word identical in February and March, that text is not
        boilerplate; it is a real situational report that happens not to have
        changed, and it must stay independently retrievable at BOTH dates or
        a March-filtered query returns nothing.

        So two different rules:

        BOILERPLATE, dropped. Repeats across >= N documents AND carries no
        figures. "OCHA coordinates the global emergency response" has no
        numbers in it; a cluster update always does. That one extra predicate
        is what separates standing furniture from unchanged substance.

        UNCHANGED CONTENT, kept. Identical text across editions is retained
        per edition and tagged with a shared `content_group`, so the
        retriever can collapse repeats in the RESULT SET while the index
        keeps every date addressable. Redundancy is a ranking problem, not a
        storage problem — solving it by deletion loses information that
        cannot be recovered at query time.

        Only within-document repeats are truly dropped.
        """
        threshold = self.dedup_cfg.get("boilerplate_doc_threshold", 5)
        needs_no_figures = self.dedup_cfg.get("boilerplate_requires_no_figures", True)

        docs_per_hash: dict[str, set[int]] = {}
        for c in chunks:
            docs_per_hash.setdefault(c.content_sha256, set()).add(c.report_id)

        seen_in_doc: set[tuple[int, str]] = set()
        for c in chunks:
            n_docs = len(docs_per_hash[c.content_sha256])
            c.content_group = c.content_sha256 if n_docs > 1 else None
            c.n_editions = n_docs

            # Two independent guards. A chunk carrying figures is substance,
            # not furniture; so is a chunk that landed in a recognised cluster
            # section. Standing text like "About OCHA" has neither.
            looks_substantive = (needs_no_figures and c.contains_figures) or c.sector
            if c.flags and "furniture" in c.flags:
                looks_substantive = False
            if (n_docs >= threshold or "furniture" in c.flags) and not looks_substantive:
                c.is_boilerplate = True
                c.flags.append("boilerplate")

            key = (c.report_id, c.content_sha256)
            if self.dedup_cfg.get("exact_hash", True):
                if key in seen_in_doc:
                    c.duplicate_of = c.chunk_id  # repeat within one document
                    c.flags.append("duplicate_in_doc")
                else:
                    seen_in_doc.add(key)
        return chunks


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(cfg: dict[str, Any]) -> dict[str, Any]:
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    chunker = SitrepChunker(cfg)
    docs = [json.loads(l) for l in
            (interim / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))
    chunker.mark_duplicates(chunks)

    keep = [c for c in chunks if not c.duplicate_of and
            not (c.is_boilerplate and cfg["dedup"].get("drop_boilerplate"))]

    out = processed / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for c in keep:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    if cfg["chunking"].get("emit_parent_sections"):
        parents: dict[str, dict] = {}
        for c in chunks:
            p = parents.setdefault(c.parent_id, {
                "parent_id": c.parent_id, "report_id": c.report_id,
                "heading": c.heading, "sector": c.sector,
                "date_original": c.date_original, "url": c.url,
                "title": c.title, "bodies": []})
            if not c.duplicate_of:
                p["bodies"].append(c.body)
        with (processed / "parents.jsonl").open("w", encoding="utf-8") as fh:
            for p in parents.values():
                p["text"] = "\n\n".join(p.pop("bodies"))
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    stats = summarize(chunks, keep, docs)
    (processed / "chunk_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def summarize(all_chunks, kept, docs) -> dict[str, Any]:
    toks = sorted(c.n_tokens for c in kept) or [0]
    return {
        "documents": len(docs),
        "chunks_built": len(all_chunks),
        "chunks_kept": len(kept),
        "dropped_duplicate": sum(1 for c in all_chunks if c.duplicate_of),
        "dropped_boilerplate": sum(1 for c in all_chunks
                                   if c.is_boilerplate and not c.duplicate_of),
        "chunks_per_doc": round(len(kept) / max(len(docs), 1), 1),
        "tokens": {
            "p10": toks[len(toks) // 10], "p50": toks[len(toks) // 2],
            "p90": toks[min(len(toks) * 9 // 10, len(toks) - 1)], "max": toks[-1],
        },
        "over_budget": sum(1 for c in kept if "over_budget" in c.flags),
        "with_table": sum(1 for c in kept if c.contains_table),
        "with_figures": sum(1 for c in kept if c.contains_figures),
        "unsectioned": sum(1 for c in kept if not c.sector),
        "by_sector": dict(Counter(c.sector or "(none)" for c in kept).most_common()),
    }


if __name__ == "__main__":
    import argparse

    from .ingest import load_config, quiet_logs

    ap = argparse.ArgumentParser(prog="src.chunker")
    ap.add_argument("--config", default="config/corpus.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    quiet_logs()
    print(json.dumps(run(load_config(a.config)), indent=2))


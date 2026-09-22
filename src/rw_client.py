"""
ReliefWeb API v2 client.

Design notes
------------
appname
    Mandatory since 1 Nov 2025. Requests without a pre-approved appname are
    rejected. We fail loudly at construction rather than letting a 400 show
    up 200 documents into a run.

POST over GET
    The v2 API accepts the same query as a JSON body on POST. Compound
    filters (AND of country / format / language / date-range) are far more
    readable as nested JSON than as
    `filter[conditions][0][field]=...&filter[conditions][1][value][from]=...`
    and there is no URL-length ceiling to worry about.

Pagination
    `limit` is capped at 1000 rows per call. Deep `offset` paging against an
    Elasticsearch-backed API is fragile past ~10k, so we slice the date range
    into calendar windows and page within each window. Each window asserts
    that totalCount < the offset ceiling; if a window is too dense, it is
    subdivided rather than silently truncated.

Quota
    1000 calls/day. The counter is persisted to disk so it survives process
    restarts, which is the only way it is useful — a crashed run that already
    spent 600 calls still spent them.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

log = logging.getLogger(__name__)

API_ROOT = "https://api.reliefweb.int/v2"
MAX_LIMIT = 1000          # hard API cap on rows per call
MAX_OFFSET = 10_000       # practical ES deep-paging ceiling
DAILY_CALL_BUDGET = 1000


class QuotaExceeded(RuntimeError):
    """Raised when the daily call budget is exhausted."""


class WindowTooDense(RuntimeError):
    """Raised when a date window holds more rows than offset paging can reach."""


@dataclass
class QuotaLedger:
    """Disk-backed counter of API calls made today (UTC)."""

    path: Path
    budget: int = DAILY_CALL_BUDGET
    _state: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        today = datetime.now(timezone.utc).date().isoformat()
        if self._state.get("date") != today:
            self._state = {"date": today, "calls": 0}
            self._flush()

    @property
    def used(self) -> int:
        return self._state["calls"]

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def spend(self, n: int = 1) -> None:
        if self.remaining < n:
            raise QuotaExceeded(
                f"Daily ReliefWeb budget exhausted ({self.used}/{self.budget}). "
                "Re-run after 00:00 UTC; the manifest will resume where it stopped."
            )
        self._state["calls"] += n
        self._flush()
        if self.remaining in (200, 100, 50, 10):
            log.warning("ReliefWeb quota low: %d calls remaining today", self.remaining)

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._state))


class ReliefWebClient:
    def __init__(
        self,
        appname: str | None = None,
        quota_path: str | Path = "data/.rw_quota.json",
        timeout: int = 60,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.appname = appname or os.environ.get("RELIEFWEB_APPNAME", "").strip()
        if not self.appname:
            raise ValueError(
                "No ReliefWeb appname. Since 1 Nov 2025 the API rejects requests "
                "without a pre-approved appname. Request one via the form linked "
                "from https://apidoc.reliefweb.int/ (format: org-purpose-random, "
                "e.g. 'unfpa-som-sitrep-rag-7f3a9c'), then export RELIEFWEB_APPNAME."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.quota = QuotaLedger(Path(quota_path))
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": f"{self.appname} (+reliefweb-somalia-rag)"}
        )

    # -- transport ----------------------------------------------------------

    def _post(self, resource: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_ROOT}/{resource}?appname={self.appname}"
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            self.quota.spend(1)
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("network error (%s), retry %d", exc, attempt)
                time.sleep(delay + random.random())
                delay *= 2
                continue

            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries:
                    r.raise_for_status()
                wait = float(r.headers.get("Retry-After", delay)) + random.random()
                log.warning("HTTP %d, backing off %.1fs", r.status_code, wait)
                time.sleep(wait)
                delay *= 2
                continue
            if r.status_code in (400, 403):
                raise RuntimeError(
                    f"HTTP {r.status_code} from ReliefWeb — most often an "
                    f"unapproved appname or a malformed filter. Body: {r.text[:400]}"
                )
            r.raise_for_status()
        raise RuntimeError("unreachable")

    # -- query construction -------------------------------------------------

    @staticmethod
    def build_filter(
        country_iso3: str,
        formats: list[str],
        language_code: str,
        status: str,
        date_field: str,
        window_from: date | None = None,
        window_to: date | None = None,
        sources: list[str] | None = None,
        source_field: str = "source.shortname",
    ) -> dict[str, Any]:
        """Compound AND filter. Date bounds are inclusive at both ends.

        `sources` is optional and omitted entirely when empty — an empty
        value array is not the same as no condition, and the API will
        happily return zero rows for one while you assume the other.
        """
        conditions: list[dict[str, Any]] = [
            {"field": "country.iso3", "value": country_iso3.lower()},
            {"field": "format.name", "value": formats, "operator": "OR"},
            {"field": "language.code", "value": language_code},
            {"field": "status", "value": status},
        ]
        if sources:
            conditions.append(
                {"field": source_field, "value": sources, "operator": "OR"}
            )
        if window_from and window_to:
            conditions.append({
                "field": date_field,
                "value": {
                    "from": f"{window_from.isoformat()}T00:00:00+00:00",
                    "to": f"{window_to.isoformat()}T23:59:59+00:00",
                },
            })
        return {"operator": "AND", "conditions": conditions}

    def facet(
        self, filters: dict[str, Any], field_name: str, limit: int = 50
    ) -> list[tuple[str, int]]:
        """Value counts for one field, no rows returned. Costs 1 call.

        This is how you find out what is actually in the corpus before
        spending a fetch on it — publisher names, formats, years.

        The response is unwrapped defensively. Facet envelopes differ between
        API versions and between named and unnamed facets, and a KeyError on
        the very first live command is a bad way to learn that. If the shape
        is unfamiliar the raw keys are reported instead of raising.
        """
        res = self._post("reports", {
            "filter": filters,
            "facets": [{"field": field_name, "name": "f", "limit": limit}],
            "limit": 0,
        })
        return _parse_facet(res, field_name)

    # -- public API ---------------------------------------------------------

    def count(self, filters: dict[str, Any]) -> int:
        """Row count for a filter without pulling any rows (costs 1 call)."""
        res = self._post("reports", {"filter": filters, "limit": 0})
        return int(res.get("totalCount", 0))

    def iter_reports(
        self,
        filters: dict[str, Any],
        fields: list[str],
        page_size: int = MAX_LIMIT,
        sort: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every report matching `filters`, paging by offset."""
        page_size = min(page_size, MAX_LIMIT)
        total = self.count(filters)
        if total > MAX_OFFSET:
            raise WindowTooDense(f"{total} rows exceeds offset ceiling {MAX_OFFSET}")
        log.info("window holds %d reports", total)

        offset = 0
        seen = 0
        while offset < total:
            res = self._post(
                "reports",
                {
                    "filter": filters,
                    "fields": {"include": fields},
                    "limit": page_size,
                    "offset": offset,
                    # Stable sort. Sorting by a mutable field (date.changed)
                    # while paging can duplicate or skip rows mid-run.
                    "sort": sort or ["id:asc"],
                },
            )
            data = res.get("data", [])
            if not data:
                break
            for item in data:
                seen += 1
                yield self._flatten(item)
            offset += len(data)
        if seen != total:
            log.warning("expected %d reports, yielded %d", total, seen)

    def iter_windows(
        self,
        *,
        country_iso3: str,
        formats: list[str],
        language_code: str,
        status: str,
        date_field: str,
        date_from: date,
        date_to: date,
        fields: list[str],
        interval: str = "year",
        sources: list[str] | None = None,
        source_field: str = "source.shortname",
    ) -> Iterator[dict[str, Any]]:
        """Walk the date range window by window, subdividing dense windows."""
        def _f(lo: date, hi: date) -> dict[str, Any]:
            return self.build_filter(
                country_iso3, formats, language_code, status, date_field,
                lo, hi, sources=sources, source_field=source_field,
            )

        for lo, hi in _calendar_windows(date_from, date_to, interval):
            try:
                yield from self.iter_reports(_f(lo, hi), fields)
            except WindowTooDense:
                log.info("subdividing dense window %s..%s", lo, hi)
                for sub_lo, sub_hi in _calendar_windows(lo, hi, "month"):
                    yield from self.iter_reports(_f(sub_lo, sub_hi), fields)

    # -- shaping ------------------------------------------------------------

    @staticmethod
    def _flatten(item: dict[str, Any]) -> dict[str, Any]:
        """Lift `fields` up and normalise the bits the pipeline depends on."""
        f = item.get("fields", {})
        files = [
            {
                "id": x.get("id"),
                "url": x.get("url"),
                "filename": x.get("filename"),
                "mimetype": x.get("mimetype"),
                "description": x.get("description"),
            }
            for x in _as_list(f.get("file"))
        ]
        sources = _as_list(f.get("source"))
        primary = (_as_list(f.get("primary_country")) or [{}])[0]
        date_f = f.get("date") or {}
        return {
            "report_id": int(item.get("id") or f.get("id")),
            "title": f.get("title"),
            "body": f.get("body") or "",
            "url": f.get("url") or f.get("url_alias"),
            "origin": f.get("origin"),
            "date_original": date_f.get("original"),
            "date_created": date_f.get("created"),
            "date_changed": date_f.get("changed"),
            "sources": [s.get("name") for s in sources if s.get("name")],
            "source_shortnames": [s.get("shortname") for s in sources
                                  if s.get("shortname")],
            "source_types": [t.get("name") for s in sources
                             for t in _as_list(s.get("type")) if t.get("name")],
            "primary_country": primary.get("name"),
            "primary_country_iso3": primary.get("iso3"),
            "is_primary_country_som": (primary.get("iso3") or "").lower() == "som",
            "countries": [c.get("name") for c in _as_list(f.get("country"))],
            "formats": [x.get("name") for x in _as_list(f.get("format"))],
            "themes": [x.get("name") for x in _as_list(f.get("theme"))],
            "disasters": [x.get("name") for x in _as_list(f.get("disaster"))],
            "glides": [x.get("glide") for x in _as_list(f.get("disaster"))],
            "ocha_products": [x.get("name") for x in _as_list(f.get("ocha_product"))],
            "files": files,
            "body_chars": len(f.get("body") or ""),
        }


def _as_list(v: Any) -> list:
    """Normalise a ReliefWeb field to a list.

    The API returns single-valued fields as bare objects and multi-valued
    ones as arrays, and which is which is not guessable from the field name:
    `country` is an array, `primary_country` is an object. Assuming a list
    everywhere raises KeyError: 0 three hundred documents into a fetch, which
    is an expensive way to find out. Assuming a dict everywhere silently
    keeps only the first source. Normalising once here removes the question.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return [v] if isinstance(v, dict) else []


def _calendar_windows(
    start: date, end: date, interval: str
) -> Iterator[tuple[date, date]]:
    """Inclusive calendar windows covering [start, end]."""
    if interval not in {"year", "month"}:
        raise ValueError(f"unsupported interval: {interval}")
    cur = start
    while cur <= end:
        if interval == "year":
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = (
                date(cur.year + 1, 1, 1)
                if cur.month == 12
                else date(cur.year, cur.month + 1, 1)
            )
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def _parse_facet(res: dict[str, Any], field_name: str) -> list[tuple[str, int]]:
    """Pull (value, count) pairs out of whichever facet envelope came back."""
    facets = (res.get("embedded", {}) or {}).get("facets") or res.get("facets") or {}

    block: Any = None
    if isinstance(facets, dict):
        # Keyed by the facet's `name`, else by the field, else take the only one.
        block = facets.get("f") or facets.get(field_name)
        if block is None and len(facets) == 1:
            block = next(iter(facets.values()))
    elif isinstance(facets, list):
        block = next((f for f in facets
                      if f.get("name") in ("f", field_name)), None) or \
                (facets[0] if facets else None)

    data = block.get("data", block) if isinstance(block, dict) else block
    if not isinstance(data, list):
        log.error("unrecognised facet envelope; top-level keys=%s facet keys=%s",
                  list(res.keys()), list(facets) if hasattr(facets, "__iter__") else facets)
        return []

    out = []
    for b in data:
        if not isinstance(b, dict):
            continue
        value = b.get("value") or b.get("name") or b.get("key")
        count = b.get("count", b.get("doc_count", 0))
        if value is not None:
            out.append((str(value), int(count)))
    return sorted(out, key=lambda x: -x[1])

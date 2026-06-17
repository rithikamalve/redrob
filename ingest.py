"""
ingest.py — Candidate data loading and normalisation.

Public API
----------
    stream_candidates(path) -> Iterator[dict]   memory-efficient, one record at a time
    load_candidates(path)   -> list[dict]        all records in memory

Every record yielded/returned has already been processed by normalise_candidate():
  - Future dates clamped to DATE_TODAY
  - _days_inactive (int) pre-computed
  - _activity_reliable (bool) flag set
  - career_history sorted most-recent-first
  - salary min/max swapped if inverted
  - profile._location_norm set to canonical city name (lowercase)

Supports both plain .jsonl and gzipped .jsonl.gz files transparently.
"""
from __future__ import annotations

import datetime
import gzip
import json
import logging
from typing import IO, Iterator, List, Optional

from config import CITY_CANON, DATE_TODAY
from scorer.utils import parse_date

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_jsonl(path: str) -> IO[str]:
    """Return a text-mode file handle for .jsonl or .jsonl.gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _clamp_date_str(date_str: Optional[str]) -> str:
    """
    Clamp a date string to DATE_TODAY if it is in the future.
    Returns the original string (or empty string) if it cannot be parsed.
    """
    if not date_str:
        return date_str or ""
    parsed = parse_date(date_str)
    if parsed is None:
        return date_str
    return min(parsed, DATE_TODAY).isoformat()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_candidate(candidate: dict) -> dict:
    """
    Apply all pre-processing normalisations to a candidate dict **in place**.

    Mutations:
      signals["last_active_date"]               clamped to DATE_TODAY
      signals["signup_date"]                    clamped to DATE_TODAY
      signals["expected_salary_range_inr_lpa"]  min/max swapped if inverted
      candidate["_days_inactive"]               int — days since last active
      candidate["_activity_reliable"]           bool — False if signup > last_active
      candidate["career_history"]               sorted most-recent-first by start_date
      profile["_location_norm"]                 canonical lowercased city name

    Parameters
    ----------
    candidate : raw dict from json.loads — modified in place and returned.
    """
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    # ------------------------------------------------------------------
    # 1. Clamp future dates
    # ------------------------------------------------------------------
    for field in ("last_active_date", "signup_date"):
        raw = signals.get(field)
        if raw:
            signals[field] = _clamp_date_str(raw)

    # ------------------------------------------------------------------
    # 2. days_inactive (used by honeypot Signal 5, behavioral scorer,
    #    and reasoning generator)
    # ------------------------------------------------------------------
    last_active = parse_date(signals.get("last_active_date"))
    if last_active is None:
        last_active = DATE_TODAY
    candidate["_days_inactive"] = (DATE_TODAY - last_active).days

    # ------------------------------------------------------------------
    # 3. Activity reliability flag
    #    signup_date > last_active_date is impossible — data quality issue
    # ------------------------------------------------------------------
    signup = parse_date(signals.get("signup_date"))
    candidate["_activity_reliable"] = not (
        signup is not None and signup > last_active
    )

    # ------------------------------------------------------------------
    # 4. Salary: swap min/max if inverted
    # ------------------------------------------------------------------
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    s_min = salary.get("min") or 0
    s_max = salary.get("max") or 0
    if isinstance(s_min, (int, float)) and isinstance(s_max, (int, float)):
        if s_min > s_max:
            salary["min"], salary["max"] = s_max, s_min

    # ------------------------------------------------------------------
    # 5. Sort career_history most-recent-first
    #    career_history[0] is assumed to be the current / most recent role
    #    throughout the pipeline; this sort makes that assumption safe.
    # ------------------------------------------------------------------
    career = candidate.get("career_history") or []
    career.sort(
        key=lambda r: parse_date(r.get("start_date")) or datetime.date.min,
        reverse=True,
    )

    # ------------------------------------------------------------------
    # 6. Canonical location (lowercase, aliases resolved)
    # ------------------------------------------------------------------
    raw_loc = profile.get("location", "").lower().strip()
    profile["_location_norm"] = CITY_CANON.get(raw_loc, raw_loc)

    return candidate


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def stream_candidates(path: str) -> Iterator[dict]:
    """
    Yield normalised candidate dicts one at a time from a JSONL or JSONL.GZ file.

    Memory-efficient: only one record is in memory at any point.
    Malformed JSON lines are logged and skipped without halting the stream.

    Parameters
    ----------
    path : str — path to candidates.jsonl or candidates.jsonl.gz
    """
    with _open_jsonl(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield normalise_candidate(candidate)
            except json.JSONDecodeError as exc:
                log.warning(
                    "Skipping malformed JSON at line %d: %s", line_num, exc
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Unexpected error normalising candidate at line %d: %s",
                    line_num, exc,
                    exc_info=True,
                )


def load_candidates(path: str) -> List[dict]:
    """
    Load all normalised candidates into a list.

    Use stream_candidates() instead when memory is a concern.
    For 100 K records (~465 MB uncompressed) this fits comfortably in 16 GB RAM.

    Parameters
    ----------
    path : str — path to candidates.jsonl or candidates.jsonl.gz
    """
    log.info("Loading candidates from %s …", path)
    candidates = list(stream_candidates(path))
    log.info("Loaded %d candidates.", len(candidates))
    return candidates

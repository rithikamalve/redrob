"""
scorer/utils.py — Shared date-parsing and interval utilities.

Imported by ingest, honeypot, career, logistics, and behavioral modules.
Never imports from other scorer sub-modules (no circular dependency risk).
"""
from __future__ import annotations

import datetime
import logging
from typing import List, Optional, Tuple

from config import CONSULTING_FIRMS, DATE_TODAY

log = logging.getLogger(__name__)

try:
    from dateutil import parser as _dateutil_parser
    _USE_DATEUTIL = True
except ImportError:  # pragma: no cover
    _USE_DATEUTIL = False
    log.warning("python-dateutil not installed; falling back to fromisoformat (YYYY-MM-DD only)")


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(value: Optional[str]) -> Optional[datetime.date]:
    """
    Parse an ISO-8601 date string to datetime.date.

    Returns None on any parse failure — never raises.
    Accepts None / empty string gracefully.
    """
    if not value:
        return None
    try:
        if _USE_DATEUTIL:
            return _dateutil_parser.parse(str(value)).date()
        # stdlib fallback: handles YYYY-MM-DD only
        return datetime.date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------

def months_between(start: datetime.date, end: datetime.date) -> int:
    """
    Signed integer count of whole months from start to end.
    Positive if end > start, negative if end < start.
    """
    return (end.year - start.year) * 12 + (end.month - start.month)


def is_recent(role: dict, months: int) -> bool:
    """
    Return True if a career role was active at any point within the last
    `months` months relative to DATE_TODAY.

    Roles with is_current=True or end_date=None are always considered recent.
    """
    if role.get("is_current", False):
        return True
    end = parse_date(role.get("end_date"))
    if end is None:
        return True  # treat missing end_date as current
    return months_between(end, DATE_TODAY) <= months


def sum_non_overlapping_months(
    intervals: List[Tuple[datetime.date, datetime.date]],
) -> int:
    """
    Merge overlapping (start, end) date intervals and return the total number
    of whole months covered.

    Used to avoid double-counting concurrent ML roles when computing ml_months.
    Intervals with end < start are silently dropped.
    """
    if not intervals:
        return 0

    # Drop degenerate intervals and sort by start date
    valid = [(s, e) for s, e in intervals if e >= s]
    if not valid:
        return 0

    valid.sort(key=lambda x: x[0])
    merged: List[Tuple[datetime.date, datetime.date]] = [valid[0]]

    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return sum(max(months_between(s, e), 0) for s, e in merged)


# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------

def is_consulting_company(company: str) -> bool:
    """
    Return True if the company name matches any known consulting / IT-services firm.
    Uses substring matching on the lowercased company name.
    """
    co = company.lower()
    return any(firm in co for firm in CONSULTING_FIRMS)

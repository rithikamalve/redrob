"""
scorer/career.py — Career history feature extraction.

Features computed per candidate
--------------------------------
ml_months            int    non-overlapping months in ML/AI-titled roles
product_fraction     float  fraction of career NOT at consulting firms (raw, no discount)
title_hop            bool   avg post-yr3 tenure < TITLE_HOP_MIN_TENURE_MONTHS
coding_gap           bool   most recent title is management-only (no IC keywords)
coding_gap_multiplier float 0.75 if coding_gap else 1.0
founding_team_exp    bool   any role at a startup (company_size in STARTUP_SIZES)
recent_ml            bool   ML/AI-titled role active in last RECENT_ROLE_WINDOW_MONTHS
avg_desc_words       float  avg words per career description (authenticity proxy)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import datetime

from config import (
    CODING_GAP_MULT,
    DATE_TODAY,
    EXPERIENCE_CURVE_BREAKPOINTS,
    EXPERIENCE_CURVE_DEFAULT,
    IC_KEYWORDS,
    MGMT_KEYWORDS,
    ML_AI_TITLE_KEYWORDS,
    ML_MONTHS_FULL_SCORE,
    RECENT_ROLE_WINDOW_MONTHS,
    STARTUP_SIZES,
    TITLE_HOP_GRACE_DAYS,
    TITLE_HOP_MIN_TENURE_MONTHS,
)
from scorer.utils import (
    is_consulting_company,
    is_recent,
    parse_date,
    sum_non_overlapping_months,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experience curve
# ---------------------------------------------------------------------------

def experience_curve(yoe: float) -> float:
    """
    Map years of experience to a 0–1 score.

    Peaks at 5–8 yrs (score = 1.0). Slopes off below 5 (still learning)
    and above 8 (risk of over-seniority for a founding-team IC role).

    Breakpoints match the JD's "5–9 years" guidance with slack in both directions.
    """
    if yoe < 3:    return 0.50
    if yoe < 5:    return 0.75
    if yoe <= 8:   return 1.00
    if yoe <= 10:  return 0.90
    if yoe <= 12:  return 0.80
    return EXPERIENCE_CURVE_DEFAULT


# ---------------------------------------------------------------------------
# ML months helper (re-exported from utils so callers can import it directly)
# ---------------------------------------------------------------------------

# sum_non_overlapping_months is imported from scorer.utils; re-used below.


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_career_features(
    career_history: List[Dict[str, Any]],
    yoe: float,
) -> Dict[str, Any]:
    """
    Extract all career features from a candidate's career_history.

    Precondition: career_history is sorted most-recent-first (done in ingest).

    Parameters
    ----------
    career_history : list of role dicts, already sorted
    yoe            : profile["years_of_experience"]

    Returns
    -------
    dict with keys:
        ml_months, product_fraction, title_hop, coding_gap,
        coding_gap_multiplier, founding_team_exp, recent_ml, avg_desc_words
    """
    if not career_history:
        return {
            "ml_months":             0,
            "product_fraction":      0.5,
            "title_hop":             False,
            "coding_gap":            False,
            "coding_gap_multiplier": 1.0,
            "founding_team_exp":     False,
            "recent_ml":             False,
            "avg_desc_words":        0.0,
        }

    # ------------------------------------------------------------------
    # 1. ML months — non-overlapping union of ML/AI-titled role intervals
    # ------------------------------------------------------------------
    ml_intervals: List[Tuple[datetime.date, datetime.date]] = []
    for role in career_history:
        title = (role.get("title") or "").lower()
        if not any(kw in title for kw in ML_AI_TITLE_KEYWORDS):
            continue
        start = parse_date(role.get("start_date"))
        if start is None:
            continue
        end_raw = role.get("end_date")
        end = parse_date(end_raw) if end_raw else DATE_TODAY
        if end is None:
            end = DATE_TODAY
        ml_intervals.append((start, end))

    ml_months = sum_non_overlapping_months(ml_intervals)

    # ------------------------------------------------------------------
    # 2. Product fraction — career time NOT at consulting firms (raw, no discount)
    # ------------------------------------------------------------------
    total_months = sum(float(r.get("duration_months") or 0) for r in career_history)
    consulting_mo = sum(
        float(r.get("duration_months") or 0)
        for r in career_history
        if is_consulting_company(r.get("company") or "")
    )
    product_fraction = (
        (total_months - consulting_mo) / total_months
        if total_months > 0 else 0.5
    )

    # ------------------------------------------------------------------
    # 3. Title hopping — flag only for post-year-3 roles
    #    Early-career switching is normal; this only fires after stabilisation.
    # ------------------------------------------------------------------
    valid_starts = [
        (parse_date(r.get("start_date")), r)
        for r in career_history
        if parse_date(r.get("start_date")) is not None
    ]

    if len(valid_starts) < 2:
        title_hop = False
    else:
        career_start = min(s for s, _ in valid_starts)
        post_yr3_roles = [
            r for s, r in valid_starts
            if (s - career_start).days > TITLE_HOP_GRACE_DAYS
        ]
        if len(post_yr3_roles) > 1:
            avg_tenure = (
                sum(float(r.get("duration_months") or 0) for r in post_yr3_roles)
                / len(post_yr3_roles)
            )
            title_hop = avg_tenure < TITLE_HOP_MIN_TENURE_MONTHS
        else:
            title_hop = False

    # ------------------------------------------------------------------
    # 4. Coding gap — most recent title is management-only (no IC keywords)
    #    career_history[0] is the most recent role (sorted in ingest).
    # ------------------------------------------------------------------
    recent_title = (career_history[0].get("title") or "").lower()
    is_mgmt = any(kw in recent_title for kw in MGMT_KEYWORDS)
    is_ic   = any(kw in recent_title for kw in IC_KEYWORDS)
    coding_gap = is_mgmt and not is_ic
    coding_gap_multiplier = CODING_GAP_MULT if coding_gap else 1.0

    # ------------------------------------------------------------------
    # 5. Startup / founding-team exposure
    # ------------------------------------------------------------------
    founding_team_exp = any(
        r.get("company_size") in STARTUP_SIZES for r in career_history
    )

    # ------------------------------------------------------------------
    # 6. Recent ML role (last RECENT_ROLE_WINDOW_MONTHS months)
    # ------------------------------------------------------------------
    recent_ml = any(
        any(kw in (r.get("title") or "").lower() for kw in ML_AI_TITLE_KEYWORDS)
        for r in career_history
        if is_recent(r, RECENT_ROLE_WINDOW_MONTHS)
    )

    # ------------------------------------------------------------------
    # 7. Average description word count (authenticity / depth proxy)
    # ------------------------------------------------------------------
    word_counts = [
        len((r.get("description") or "").split())
        for r in career_history
    ]
    avg_desc_words = sum(word_counts) / max(len(word_counts), 1)

    return {
        "ml_months":             ml_months,
        "product_fraction":      float(product_fraction),
        "title_hop":             title_hop,
        "coding_gap":            coding_gap,
        "coding_gap_multiplier": coding_gap_multiplier,
        "founding_team_exp":     founding_team_exp,
        "recent_ml":             recent_ml,
        "avg_desc_words":        avg_desc_words,
    }

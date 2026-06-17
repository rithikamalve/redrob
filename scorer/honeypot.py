"""
scorer/honeypot.py — Internal consistency checks for honeypot detection.

Five signals are computed per candidate.
  ≥ 2 signals fired  →  honeypot = True        (hard disqualify in rank.py)
  == 1 signal fired  →  honeypot_soft = True    (HONEYPOT_SOFT_MULT = 0.65×)
     0 signals fired  →  clean

Signal definitions
------------------
  1. timeline_contradiction — stated vs actual role duration disagree by > 18 months
                              in ≥ 2 roles
  2. expert_zero_duration   — ≥ 5 skills claimed as "expert" with 0 months used
  3. yoe_vs_graduation      — years_of_experience exceeds what graduation year permits
  4. implausible_seniority  — VP / Director / CTO title with < 4 years experience
  5. perfect_and_stale      — 100 % profile completeness + inactive for > 2 years

Design rules:
  - Missing data never fires a signal (missing ≠ impossible).
  - Each signal requires unambiguous impossibility, not mere suspicion.
  - career_history must be sorted most-recent-first before calling compute_honeypot()
    (done by ingest.normalise_candidate).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from config import (
    DATE_TODAY,
    EXPERT_ZERO_DUR_MIN_SKILLS,
    HONEYPOT_HARD_SIGNAL_COUNT,
    PERFECT_COMPLETENESS,
    PERFECT_STALE_INACTIVE_DAYS,
    SENIORITY_MAX_YOE,
    SENIORITY_TITLE_KEYWORDS,
    TIMELINE_CONTRADICTION_MONTHS,
    YOE_GRAD_SLACK_YEARS,
)
from scorer.utils import months_between, parse_date

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal 1 — Career timeline contradiction
# ---------------------------------------------------------------------------

def _signal_timeline_contradiction(career_history: List[Dict[str, Any]]) -> bool:
    """
    Return True if ≥ 2 roles have |stated_duration_months − actual_months| > 18.

    actual_months is derived independently from start_date / end_date, exposing
    cases where the stated duration was fabricated.

    Note on current roles (end_date=None):
        We use DATE_TODAY as the end, so actual_months grows with time.
        A current role might show a discrepancy if the dataset is old, but
        ≥ 2 simultaneous contradictions in a fresh dataset strongly indicates
        fabrication. Single-role discrepancies only trigger honeypot_soft.
    """
    contradictions = 0
    for role in career_history:
        start = parse_date(role.get("start_date"))
        if start is None:
            continue  # unparseable start date — skip, not penalise

        end_raw = role.get("end_date")
        end = parse_date(end_raw) if end_raw else DATE_TODAY
        if end is None:
            end = DATE_TODAY

        actual = months_between(start, end)
        stated = int(role.get("duration_months") or 0)

        if abs(stated - actual) > TIMELINE_CONTRADICTION_MONTHS:
            contradictions += 1
            if contradictions >= 2:
                return True  # short-circuit — no need to check remaining roles

    return False


# ---------------------------------------------------------------------------
# Signal 2 — Expert skills never used
# ---------------------------------------------------------------------------

def _signal_expert_zero_duration(skills: List[Dict[str, Any]]) -> bool:
    """
    Return True if ≥ EXPERT_ZERO_DUR_MIN_SKILLS skills are claimed as "expert"
    with duration_months == 0.

    Genuine expertise accumulates months of use.  Mass-claiming expert status
    with zero usage time is a synthetic-data fingerprint.
    """
    count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months") or 0) == 0
    )
    return count >= EXPERT_ZERO_DUR_MIN_SKILLS


# ---------------------------------------------------------------------------
# Signal 3 — YOE vs graduation year
# ---------------------------------------------------------------------------

def _signal_yoe_vs_graduation(candidate: Dict[str, Any]) -> bool:
    """
    Return True if years_of_experience exceeds what the earliest graduation
    year permits (with YOE_GRAD_SLACK_YEARS years of slack for gap years /
    PhDs / early employment).

    Only fires if education is non-empty.
    Empty education list → signal does NOT fire (missing data ≠ impossible).
    """
    education = candidate.get("education") or []
    if not education:
        return False

    grad_years = [
        int(e["end_year"])
        for e in education
        if isinstance(e.get("end_year"), (int, float)) and int(e["end_year"]) > 1970
    ]
    if not grad_years:
        return False

    min_grad_year = min(grad_years)
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    max_plausible_yoe = (DATE_TODAY.year - min_grad_year) + YOE_GRAD_SLACK_YEARS
    return yoe > max_plausible_yoe


# ---------------------------------------------------------------------------
# Signal 4 — Implausible seniority
# ---------------------------------------------------------------------------

def _signal_implausible_seniority(candidate: Dict[str, Any]) -> bool:
    """
    Return True if the candidate holds a VP / Director / CTO / Chief title
    with fewer than SENIORITY_MAX_YOE (4) years of total experience.

    Checks both profile.current_title and the most recent career_history entry
    (career_history must already be sorted most-recent-first).
    """
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    if yoe >= SENIORITY_MAX_YOE:
        return False  # experience is plausible — skip keyword check

    titles: List[str] = [
        candidate.get("profile", {}).get("current_title", "").lower(),
    ]
    career = candidate.get("career_history") or []
    if career:
        titles.append(career[0].get("title", "").lower())

    return any(kw in title for title in titles for kw in SENIORITY_TITLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Signal 5 — Perfect-and-abandoned profile
# ---------------------------------------------------------------------------

def _signal_perfect_and_stale(
    candidate: Dict[str, Any],
    days_inactive: int,
) -> bool:
    """
    Return True if profile_completeness_score == 100 AND the candidate has
    been inactive for more than PERFECT_STALE_INACTIVE_DAYS (730) days.

    A fully complete profile implies an active job-seeker; two years of
    inactivity combined with perfect completeness is inconsistent.
    """
    completeness = float(
        candidate.get("redrob_signals", {}).get("profile_completeness_score") or 0
    )
    return (
        completeness >= PERFECT_COMPLETENESS
        and days_inactive > PERFECT_STALE_INACTIVE_DAYS
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_honeypot(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all five honeypot signals against a single normalised candidate.

    Preconditions (guaranteed by ingest.normalise_candidate):
      - candidate["career_history"] is sorted most-recent-first
      - candidate["_days_inactive"] (int) is present

    Returns
    -------
    dict
        honeypot       : bool  — True if signal_count ≥ HONEYPOT_HARD_SIGNAL_COUNT (2)
        honeypot_soft  : bool  — True if signal_count == 1
        signal_count   : int   — 0–5
        signals_fired  : list[str] — names of fired signals (for audit/debug)
    """
    career_history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    days_inactive = int(candidate.get("_days_inactive") or 0)

    fired: List[str] = []

    if _signal_timeline_contradiction(career_history):
        fired.append("timeline_contradiction")

    if _signal_expert_zero_duration(skills):
        fired.append("expert_zero_duration")

    if _signal_yoe_vs_graduation(candidate):
        fired.append("yoe_vs_graduation")

    if _signal_implausible_seniority(candidate):
        fired.append("implausible_seniority")

    if _signal_perfect_and_stale(candidate, days_inactive):
        fired.append("perfect_and_stale")

    count = len(fired)

    if count >= 1:
        log.debug(
            "Candidate %s — %d honeypot signal(s): %s",
            candidate.get("candidate_id", "?"),
            count,
            fired,
        )

    return {
        "honeypot":      count >= HONEYPOT_HARD_SIGNAL_COUNT,
        "honeypot_soft": count == 1,
        "signal_count":  count,
        "signals_fired": fired,
    }

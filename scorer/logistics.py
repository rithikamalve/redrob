"""
scorer/logistics.py — Location classification, consulting fraction,
domain-mismatch gate, salary fit, and work-mode scoring.

Hard gates (hard_out = True) are stored as a flag in features.parquet and
applied as hard filters in rank.py *before* composite scoring runs.
Candidates that are hard_out never appear in the output.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from config import (
    CONSULTING_FIRMS,
    CONSULTING_HARD_OUT_FRACTION,
    CONSULTING_MULTIPLIER_THRESHOLD,
    CONSULTING_PRODUCT_PROXY_WORDS,
    CV_SPEECH_KEYWORDS,
    JD_SALARY_MAX,
    JD_SALARY_MIN,
    LOCATION_SCORES,
    NLP_IR_KEYWORDS,
    RECENT_ROLE_WINDOW_MONTHS,
    SALARY_NO_OVERLAP_SCORE,
    SALARY_UNDISCLOSED_SCORE,
    TIER1_INDIA_CITIES,
    TIER2_INDIA_CITIES,
    WORKMODE_SCORES,
)
from scorer.utils import is_consulting_company, is_recent, parse_date

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate A — Location classification
# ---------------------------------------------------------------------------

def classify_location(candidate: Dict[str, Any]) -> str:
    """
    Classify a candidate's location into one of five tiers:

    'tier1_india'  — major Indian metros mentioned in the JD
    'tier2_india'  — other Indian cities
    'remote_india' — India-based, listed as remote
    'intl_open'    — outside India, willing to relocate
    'intl_no'      — outside India, not willing to relocate → hard_out

    Uses profile._location_norm (set by ingest.normalise_candidate).
    """
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    country = (profile.get("country") or "").lower().strip()
    # _location_norm is the canonical lowercase city name set by ingest
    location = (profile.get("_location_norm") or profile.get("location") or "").lower().strip()
    relocate = bool(signals.get("willing_to_relocate", False))

    if country != "india":
        return "intl_open" if relocate else "intl_no"

    if "remote" in location:
        return "remote_india"

    if any(city in location for city in TIER1_INDIA_CITIES):
        return "tier1_india"

    if any(city in location for city in TIER2_INDIA_CITIES):
        return "tier2_india"

    # Unknown Indian city — treat conservatively as tier2
    return "tier2_india"


# ---------------------------------------------------------------------------
# Gate B — Consulting fraction + multiplier
# ---------------------------------------------------------------------------

def consulting_fraction(career_history: List[Dict[str, Any]]) -> float:
    """
    Fraction of total career months spent at consulting / IT-services firms,
    in [0.0, 1.0].

    Roles with description language suggesting genuine product-embedded work
    receive a 0.6× discount on their consulting months — this avoids penalising
    candidates who were placed as dedicated engineers at a product client.
    """
    total_months = sum(float(r.get("duration_months") or 0) for r in career_history)
    if total_months == 0:
        return 0.0

    consulting_months = 0.0
    for role in career_history:
        if not is_consulting_company(role.get("company") or ""):
            continue
        months = float(role.get("duration_months") or 0)
        desc = (role.get("description") or "").lower()
        if any(proxy in desc for proxy in CONSULTING_PRODUCT_PROXY_WORDS):
            months *= 0.6
        consulting_months += months

    return min(consulting_months / total_months, 1.0)


def compute_consulting_multiplier(fraction: float) -> float:
    """
    Map consulting fraction to a composite score multiplier.

    fraction >= 1.0          → 0.0  (caller also sets hard_out = True)
    fraction in (0.5, 1.0)  → linear: 0.51 → ~1.0×,  0.99 → ~0.51×
    fraction <= 0.5          → 1.0  (no penalty)
    """
    if fraction >= CONSULTING_HARD_OUT_FRACTION:
        return 0.0
    if fraction > CONSULTING_MULTIPLIER_THRESHOLD:
        return 1.0 - (fraction - CONSULTING_MULTIPLIER_THRESHOLD)
    return 1.0


# ---------------------------------------------------------------------------
# Gate C — Domain mismatch
# ---------------------------------------------------------------------------

def domain_mismatch(career_history: List[Dict[str, Any]]) -> bool:
    """
    Return True if the candidate's recent work is dominated by computer vision /
    speech / robotics with NO NLP / IR exposure.

    Looks at roles active in the last RECENT_ROLE_WINDOW_MONTHS months.
    Falls back to the two most-recent roles if the window is empty.

    True → hard_out = True.  Does not fire on career transitions into AI.
    """
    recent_roles = [r for r in career_history if is_recent(r, RECENT_ROLE_WINDOW_MONTHS)]
    if not recent_roles:
        recent_roles = career_history[:2]  # career_history is sorted most-recent-first
    if not recent_roles:
        return False

    all_text = " ".join(
        (r.get("description") or "") + " " + (r.get("title") or "")
        for r in recent_roles
    ).lower()

    has_cv_speech = any(kw in all_text for kw in CV_SPEECH_KEYWORDS)
    has_nlp_ir = any(kw in all_text for kw in NLP_IR_KEYWORDS)

    return has_cv_speech and not has_nlp_ir


# ---------------------------------------------------------------------------
# Salary fit
# ---------------------------------------------------------------------------

def salary_fit(signals: Dict[str, Any]) -> float:
    """
    Overlap score between the candidate's expected salary range and the JD range
    (JD_SALARY_MIN–JD_SALARY_MAX LPA), normalised to [0.0, 1.0].

    min == max == 0   → SALARY_UNDISCLOSED_SCORE  (neutral, not penalised)
    min == max > 0    → 0.5 if within JD range, else SALARY_NO_OVERLAP_SCORE
    No range overlap  → SALARY_NO_OVERLAP_SCORE
    Full range match  → 1.0
    """
    s = (signals.get("expected_salary_range_inr_lpa") or {})
    s_min = float(s.get("min") or 0)
    s_max = float(s.get("max") or 0)

    if s_min == 0 and s_max == 0:
        return SALARY_UNDISCLOSED_SCORE

    # Single-point expectation (after ingest swap, min <= max always holds)
    if s_min == s_max:
        return 0.5 if JD_SALARY_MIN <= s_min <= JD_SALARY_MAX else SALARY_NO_OVERLAP_SCORE

    overlap_min = max(s_min, JD_SALARY_MIN)
    overlap_max = min(s_max, JD_SALARY_MAX)

    if overlap_max < overlap_min:
        return SALARY_NO_OVERLAP_SCORE

    return min(
        (overlap_max - overlap_min) / (JD_SALARY_MAX - JD_SALARY_MIN),
        1.0,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_logistics_scores(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all logistics computations for one candidate and return a flat dict
    ready to be stored as columns in features.parquet.

    Returns
    -------
    dict with keys:
        location_class        : str
        location_score        : float
        consulting_frac       : float
        consulting_multiplier : float
        domain_mismatch       : bool
        hard_out              : bool
        salary_score          : float
        workmode_score        : float
    """
    signals = candidate.get("redrob_signals") or {}
    career_history = candidate.get("career_history") or []

    # Location
    loc_class = classify_location(candidate)
    loc_score = LOCATION_SCORES.get(loc_class, 0.0)
    hard_out_location = loc_class == "intl_no"

    # Consulting
    consult_frac = consulting_fraction(career_history)
    consult_mult = compute_consulting_multiplier(consult_frac)
    hard_out_consulting = consult_frac >= CONSULTING_HARD_OUT_FRACTION

    # Domain mismatch
    dm = domain_mismatch(career_history)

    # Combined hard gate
    hard_out = hard_out_location or hard_out_consulting or dm

    # Salary
    sal_score = salary_fit(signals)

    # Work mode — default to onsite score if unknown
    preferred_mode = (signals.get("preferred_work_mode") or "").lower()
    wm_score = WORKMODE_SCORES.get(preferred_mode, WORKMODE_SCORES["onsite"])

    return {
        "location_class":        loc_class,
        "location_score":        loc_score,
        "consulting_frac":       consult_frac,
        "consulting_multiplier": consult_mult,
        "domain_mismatch":       dm,
        "hard_out":              hard_out,
        "salary_score":          sal_score,
        "workmode_score":        wm_score,
    }

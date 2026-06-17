"""
scorer/behavioral.py — Behavioral signal feature extraction.

Converts the raw redrob_signals into normalised 0–1 scores, handling all
sentinel values and edge cases defined in the signals doc.

Sentinel values
---------------
    github_activity_score  == -1   → no GitHub linked (not zero activity)
    offer_acceptance_rate  == -1   → no offer history (not rejected all offers)
    Both treated as "unknown" and mapped to neutral scores, not penalties.

Edge cases
----------
    search_appearance_30d < MIN_SEARCH_FOR_RESPONSE  → response_rate unreliable,
        treat as RESPONSE_UNKNOWN_SCORE (0.5) rather than taking the raw value.
    avg_response_time_hours uses log-style exponential decay so that 720 hours
        is not treated as 10× worse than 72 hours on a linear scale.
    saved_by_recruiters_30d uses log1p scale so the first few saves matter more
        than the difference between 50 and 100 saves.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

from config import (
    GITHUB_UNKNOWN_SCORE,
    MIN_SEARCH_FOR_RESPONSE,
    NOTICE_LINEAR_MAX_DAYS,
    OFFER_UNKNOWN_SCORE,
    OPEN_TO_WORK_BONUS,
    RECENCY_DECAY_DAYS,
    RESPONSE_TIME_DECAY_HOURS,
    RESPONSE_UNKNOWN_SCORE,
    SAVED_LOG_MAX,
)

log = logging.getLogger(__name__)


def extract_behavioral_features(
    signals: Dict[str, Any],
    days_inactive: int,
) -> Dict[str, float]:
    """
    Derive scored behavioral features from a candidate's redrob_signals object.

    Parameters
    ----------
    signals      : candidate["redrob_signals"]
    days_inactive: pre-computed in ingest as (DATE_TODAY - last_active_date).days

    Returns
    -------
    dict with keys:
        recency_score, response_score, response_time_score, github_score,
        notice_score, saved_score, interview_score, offer_score, open_bonus
    """
    # ------------------------------------------------------------------
    # Recency — exponential decay on days since last platform activity
    # Half-life: ~90 days (0d→1.0, 90d→0.37, 180d→0.14, 365d→0.02)
    # ------------------------------------------------------------------
    recency_score = math.exp(-max(days_inactive, 0) / RECENCY_DECAY_DAYS)

    # ------------------------------------------------------------------
    # Recruiter response rate
    # Only meaningful if the candidate has had enough search exposure.
    # Below MIN_SEARCH_FOR_RESPONSE appearances → treat as unknown (neutral 0.5).
    # ------------------------------------------------------------------
    search_appearances = int(signals.get("search_appearance_30d") or 0)
    if search_appearances < MIN_SEARCH_FOR_RESPONSE:
        response_score = RESPONSE_UNKNOWN_SCORE
    else:
        response_score = float(signals.get("recruiter_response_rate") or 0)

    # ------------------------------------------------------------------
    # Response time — exponential decay (log-style)
    # 0h→1.0, 72h→0.37, 144h→0.14. Capped at 0 hours minimum.
    # ------------------------------------------------------------------
    rt = float(signals.get("avg_response_time_hours") or 0)
    response_time_score = math.exp(-max(rt, 0) / RESPONSE_TIME_DECAY_HOURS)

    # ------------------------------------------------------------------
    # GitHub activity score
    # -1 sentinel → no GitHub linked (mild negative only for ML roles; here neutral)
    # ------------------------------------------------------------------
    gh_raw = signals.get("github_activity_score")
    if gh_raw is None or gh_raw == -1:
        github_score = GITHUB_UNKNOWN_SCORE
    else:
        github_score = max(0.0, min(float(gh_raw) / 100.0, 1.0))

    # ------------------------------------------------------------------
    # Notice period — linear decay
    # 0 days→1.0, NOTICE_LINEAR_MAX_DAYS→0.0, beyond→capped at 0.0
    # 0-day notice could mean unemployed — still a positive (immediately hireable)
    # ------------------------------------------------------------------
    notice = int(signals.get("notice_period_days") or 0)
    notice_score = max(0.0, 1.0 - (notice / NOTICE_LINEAR_MAX_DAYS))

    # ------------------------------------------------------------------
    # Saved by recruiters — log1p scale
    # 0 saves→0.0, ~3 saves→0.37, SAVED_LOG_MAX saves→1.0
    # Active recruiter bookmarking is a strong demand signal.
    # ------------------------------------------------------------------
    saved = int(signals.get("saved_by_recruiters_30d") or 0)
    saved_score = min(
        math.log1p(max(saved, 0)) / math.log1p(SAVED_LOG_MAX),
        1.0,
    )

    # ------------------------------------------------------------------
    # Interview completion rate — pass-through (already 0–1)
    # ------------------------------------------------------------------
    interview_score = float(signals.get("interview_completion_rate") or 0)

    # ------------------------------------------------------------------
    # Offer acceptance rate
    # -1 sentinel → no prior offers (unknown, not zero)
    # ------------------------------------------------------------------
    offer_raw = signals.get("offer_acceptance_rate")
    if offer_raw is None or offer_raw == -1:
        offer_score = OFFER_UNKNOWN_SCORE
    else:
        offer_score = max(0.0, min(float(offer_raw), 1.0))

    # ------------------------------------------------------------------
    # Open to work bonus — flat additive bonus (applied after weighted sum)
    # Overrides stale last_active_date somewhat but recency_score still dominates.
    # ------------------------------------------------------------------
    open_bonus = OPEN_TO_WORK_BONUS if signals.get("open_to_work_flag") else 0.0

    return {
        "recency_score":       recency_score,
        "response_score":      response_score,
        "response_time_score": response_time_score,
        "github_score":        github_score,
        "notice_score":        notice_score,
        "saved_score":         saved_score,
        "interview_score":     interview_score,
        "offer_score":         offer_score,
        "open_bonus":          open_bonus,
    }

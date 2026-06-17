"""
scorer/skills.py — Skill taxonomy scoring, stuffer detection,
platform assessment bonus, education tier bonus, and technical keyword overlay.

Design notes
------------
- Each skill is matched to at most ONE taxonomy group (first match wins, break).
  This prevents double-counting skills that span multiple groups.
- Within a group, only the best-scoring skill counts (max, not sum).
  This prevents candidates from gaming scores by listing the same skill 10 times.
- Stuffer detection fires only when BOTH conditions are true: high skill count
  AND sparse career descriptions. This avoids penalising genuinely skilled people.
- assessment_bonus uses platform-verified scores, which are more reliable than
  self-reported proficiency — treated as a bonus on top of skill_score.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from config import (
    ASSESSMENT_BONUS_MAX,
    ASSESSMENT_CORE_TERMS,
    DURATION_SATURATION_MONTHS,
    EDUCATION_TIER_BONUS,
    ENDORSEMENT_BOOST_DIVISOR,
    ENDORSEMENT_BOOST_MAX,
    PROFICIENCY_WEIGHT,
    SKILL_TAXONOMY,
    SKILL_WEIGHTS,
    STUFFER_MAX_SKILLS,
    STUFFER_MIN_DESC_WORDS,
    STUFFER_MULT,
    TECH_KEYWORD_FULL_SCORE_HITS,
    TECHNICAL_KEYWORDS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy scoring
# ---------------------------------------------------------------------------

def score_skills(
    skills: List[Dict[str, Any]],
    avg_desc_words: float,
) -> Tuple[Dict[str, float], bool, float]:
    """
    Score self-reported skills against the 6 taxonomy groups.

    Scoring per skill:
        skill_score = (proficiency_weight × duration_weight) + endorsement_boost
        where:
            proficiency_weight = PROFICIENCY_WEIGHT[proficiency]
            duration_weight    = min(duration_months / 24, 1.0)
            endorsement_boost  = min(endorsements / 20, 0.2)

    Group score = max skill_score across all matching skills in that group.
    Weighted sum = Σ group_score × SKILL_WEIGHTS[group].

    Parameters
    ----------
    skills         : candidate["skills"] list
    avg_desc_words : pre-computed in extract_career_features; used for stuffer check

    Returns
    -------
    (group_scores, stuffer_flag, stuffer_multiplier)
        group_scores      : dict[group, float] — one score per taxonomy group
        stuffer_flag      : bool
        stuffer_multiplier: float
    """
    group_scores: Dict[str, float] = {g: 0.0 for g in SKILL_TAXONOMY}

    for skill in skills:
        name = (skill.get("name") or "").lower()
        if not name:
            continue

        proficiency = skill.get("proficiency") or "beginner"
        prof_w = PROFICIENCY_WEIGHT.get(proficiency, PROFICIENCY_WEIGHT["beginner"])

        duration = float(skill.get("duration_months") or 0)
        duration_w = min(duration / DURATION_SATURATION_MONTHS, 1.0)

        endorsements = float(skill.get("endorsements") or 0)
        endorsement_boost = min(endorsements / ENDORSEMENT_BOOST_DIVISOR, ENDORSEMENT_BOOST_MAX)

        skill_score = prof_w * duration_w + endorsement_boost

        for group, keywords in SKILL_TAXONOMY.items():
            if any(kw in name for kw in keywords):
                group_scores[group] = max(group_scores[group], skill_score)
                break  # each skill counts for at most one group

    # Keyword stuffer detection:
    # Fires only when both conditions hold — prevents penalising genuinely skilled people.
    stuffer_flag = (
        len(skills) > STUFFER_MAX_SKILLS
        and avg_desc_words < STUFFER_MIN_DESC_WORDS
    )
    stuffer_multiplier = STUFFER_MULT if stuffer_flag else 1.0

    if stuffer_flag:
        log.debug("Stuffer flag: %d skills, %.1f avg desc words", len(skills), avg_desc_words)

    return group_scores, stuffer_flag, stuffer_multiplier


def weighted_skill_score(group_scores: Dict[str, float]) -> float:
    """Compute weighted sum from group scores dict."""
    return sum(group_scores[g] * SKILL_WEIGHTS[g] for g in group_scores)


# ---------------------------------------------------------------------------
# Platform assessment bonus
# ---------------------------------------------------------------------------

def skill_assessment_bonus(skill_assessment_scores: Dict[str, float]) -> float:
    """
    Derive a 0–ASSESSMENT_BONUS_MAX bonus from platform-verified assessment scores.

    Only scores for skills matching ASSESSMENT_CORE_TERMS contribute.
    Returns 0.0 if the dict is empty or no relevant keys match.

    Platform-verified scores are more trustworthy than self-reported proficiency
    (a candidate cannot claim "expert" Python on a test they haven't taken).
    """
    if not skill_assessment_scores:
        return 0.0

    relevant = [
        float(v)
        for k, v in skill_assessment_scores.items()
        if any(term in k.lower() for term in ASSESSMENT_CORE_TERMS)
        and isinstance(v, (int, float))
    ]
    if not relevant:
        return 0.0

    avg_score = sum(relevant) / len(relevant)
    return min(avg_score / 100.0 * ASSESSMENT_BONUS_MAX, ASSESSMENT_BONUS_MAX)


# ---------------------------------------------------------------------------
# Education tier bonus
# ---------------------------------------------------------------------------

def education_tier_bonus(education: List[Dict[str, Any]]) -> float:
    """
    Return the highest EDUCATION_TIER_BONUS value across all education entries.

    Returns 0.0 if education is empty — missing data is not penalised.
    Relevant because this is a founding-team hire where IIT/BITS/NIT background
    is a mild positive signal in the Indian context.
    """
    if not education:
        return 0.0
    return max(
        EDUCATION_TIER_BONUS.get(e.get("tier") or "unknown", 0.0)
        for e in education
    )


# ---------------------------------------------------------------------------
# Technical keyword overlay
# ---------------------------------------------------------------------------

def tech_keyword_score(career_history: List[Dict[str, Any]]) -> float:
    """
    Scan all career description text for highly specific technical terms
    (TECHNICAL_KEYWORDS) that MiniLM may under-weight.

    These are deep-specialist terms (HNSW, ColBERT, LambdaMART, etc.) that
    a genuine expert would use naturally but a keyword stuffer is unlikely to
    include in career descriptions (as opposed to the skills list).

    TECH_KEYWORD_FULL_SCORE_HITS hits → score = 1.0.
    Returns float in [0.0, 1.0].
    """
    all_desc = " ".join(
        (r.get("description") or "").lower()
        for r in career_history
    )
    hits = sum(1 for kw in TECHNICAL_KEYWORDS if kw in all_desc)
    return min(hits / TECH_KEYWORD_FULL_SCORE_HITS, 1.0)

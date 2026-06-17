#!/usr/bin/env python3
"""
rank.py — Fast candidate ranking from pre-computed features.

Constraints: ≤5 min wall-clock, CPU only, no network.

Usage:
    python rank.py [--features features/features.parquet] [--out submission.csv]

Pipeline:
    1. Load features.parquet (written by precompute.py)
    2. Hard filter: remove hard_out and honeypot candidates
    3. Compute jd_fit / career / behavioral / logistics sub-scores (vectorized)
    4. Apply multipliers (consulting, stuffer, coding-gap, honeypot-soft)
    5. Primary composite score with BASE_WEIGHTS
    6. Sort descending; tiebreak by candidate_id ascending (validator rule)
    7. Stability check: top-20 overlap across 3 weight configs
    8. Generate reasoning for top 100
    9. Write submission CSV
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BASE_WEIGHTS,
    BEHAVIORAL_WEIGHTS,
    CAREER_WEIGHTS,
    HONEYPOT_SOFT_MULT,
    JD_FIT_WEIGHTS,
    LOGISTICS_WEIGHTS,
    ML_MONTHS_FULL_SCORE,
    RECENT_ML_BONUS_VALUE,
    RERANK_SCORE_WEIGHT,
    STARTUP_BONUS_VALUE,
    STABILITY_MIN_OVERLAP,
    STABILITY_TOP_K,
    STABILITY_WEIGHT_CONFIGS,
    TOP_N,
    TRAJECTORY_BOTH,
    TRAJECTORY_CLEAN,
    TRAJECTORY_GAP,
    TRAJECTORY_HOP,
)
from scorer.reasoning import generate_reasoning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rank")

# Skill group columns stored in parquet
_SKILL_COLS = [
    "skill_retrieval", "skill_vector_db", "skill_embeddings",
    "skill_eval_ranking", "skill_python_mlops", "skill_llm_ops",
]


# ---------------------------------------------------------------------------
# Sub-score computation
# ---------------------------------------------------------------------------

def _compute_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """Add jd_fit_score, career_score, behavioral_score, logistics_score, _total_mult."""

    # JD fit — blend bi-encoder with cross-encoder for reranked candidates
    # Candidates with rerank_score > 0 were cross-encoded; use blended score.
    # Others (not in top-RERANK_TOP_K proxy pool) fall back to bi-encoder score alone.
    # features.parquet from a precompute.py run predating the reranker won't have
    # this column at all — fall back to bi-encoder-only scoring in that case.
    if "rerank_score" in df.columns:
        has_rerank = df["rerank_score"].values > 0
        blended_semantic = np.where(
            has_rerank,
            RERANK_SCORE_WEIGHT * df["rerank_score"] + (1 - RERANK_SCORE_WEIGHT) * df["semantic_score"],
            df["semantic_score"],
        )
    else:
        log.warning(
            "'rerank_score' column missing from features.parquet — "
            "re-run precompute.py to get cross-encoder reranking. "
            "Falling back to bi-encoder-only jd_fit scoring."
        )
        blended_semantic = df["semantic_score"].values
    jd_base = (
        blended_semantic                * JD_FIT_WEIGHTS["semantic"] +
        df["weighted_skill_score"]      * JD_FIT_WEIGHTS["skills"] +
        df["tech_keyword_score"]        * JD_FIT_WEIGHTS["tech_keyword"]
    )
    df["jd_fit_score"] = (jd_base + df["skill_assessment_bonus"]).clip(0.0, 1.0)

    # Career
    ml_comp = (df["ml_months"] / ML_MONTHS_FULL_SCORE).clip(0.0, 1.0)
    traj = np.select(
        [
            df["title_hop"].values & df["coding_gap"].values,
            df["title_hop"].values,
            df["coding_gap"].values,
        ],
        [TRAJECTORY_BOTH, TRAJECTORY_HOP, TRAJECTORY_GAP],
        default=TRAJECTORY_CLEAN,
    )
    career_base = (
        ml_comp                * CAREER_WEIGHTS["ml_component"] +
        df["product_fraction"] * CAREER_WEIGHTS["product_fraction"] +
        traj                   * CAREER_WEIGHTS["trajectory"] +
        df["exp_curve_score"]  * CAREER_WEIGHTS["exp_curve"]
    )
    startup_b   = np.where(df["founding_team_exp"].values, STARTUP_BONUS_VALUE, 0.0)
    recent_ml_b = np.where(df["recent_ml"].values,         RECENT_ML_BONUS_VALUE, 0.0)
    df["career_score"] = (
        career_base + startup_b + recent_ml_b + df["education_tier_bonus"]
    ).clip(0.0, 1.0)

    # Behavioral — open_bonus is additive after weighted sum
    beh_base = (
        df["recency_score"]       * BEHAVIORAL_WEIGHTS["recency"] +
        df["response_score"]      * BEHAVIORAL_WEIGHTS["response"] +
        df["response_time_score"] * BEHAVIORAL_WEIGHTS["response_time"] +
        df["github_score"]        * BEHAVIORAL_WEIGHTS["github"] +
        df["notice_score"]        * BEHAVIORAL_WEIGHTS["notice"] +
        df["saved_score"]         * BEHAVIORAL_WEIGHTS["saved"] +
        df["interview_score"]     * BEHAVIORAL_WEIGHTS["interview"] +
        df["offer_score"]         * BEHAVIORAL_WEIGHTS["offer"]
    )
    df["behavioral_score"] = (beh_base + df["open_bonus"]).clip(0.0, 1.0)

    # Logistics
    df["logistics_score"] = (
        df["location_score"] * LOGISTICS_WEIGHTS["location"] +
        df["workmode_score"] * LOGISTICS_WEIGHTS["workmode"] +
        df["salary_score"]   * LOGISTICS_WEIGHTS["salary"]
    )

    # Combined multiplier (product of all penalty terms)
    hp_soft_mult = np.where(df["honeypot_soft"].values, HONEYPOT_SOFT_MULT, 1.0)
    df["_total_mult"] = (
        df["consulting_multiplier"].values *
        df["stuffer_multiplier"].values *
        df["coding_gap_multiplier"].values *
        hp_soft_mult
    )

    # Skill group count — for reasoning (number of non-zero taxonomy groups)
    df["skill_group_count"] = (df[_SKILL_COLS] > 0).sum(axis=1).astype(int)

    return df


def _composite(df: pd.DataFrame, weights: dict) -> pd.Series:
    base = (
        df["jd_fit_score"]     * weights["jd_fit"] +
        df["career_score"]     * weights["career"] +
        df["behavioral_score"] * weights["behavioral"] +
        df["logistics_score"]  * weights["logistics"]
    )
    return base * df["_total_mult"]


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------

def _stability_check(df: pd.DataFrame, primary_top_ids: set) -> None:
    for i, cfg in enumerate(STABILITY_WEIGHT_CONFIGS[1:], start=1):
        alt_comp    = _composite(df, cfg)
        alt_top_idx = alt_comp.nlargest(STABILITY_TOP_K).index
        alt_top_ids = set(df.loc[alt_top_idx, "candidate_id"])
        overlap     = len(primary_top_ids & alt_top_ids)
        if overlap < STABILITY_MIN_OVERLAP:
            log.warning(
                "Stability WARN config-%d: top-%d overlap=%d/%d (threshold=%d)",
                i, STABILITY_TOP_K, overlap, STABILITY_TOP_K, STABILITY_MIN_OVERLAP,
            )
        else:
            log.info("Stability OK config-%d: overlap=%d/%d", i, overlap, STABILITY_TOP_K)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates from pre-computed features")
    parser.add_argument(
        "--candidates", default=None,
        help=(
            "Path to candidates.jsonl (accepted for compatibility with the "
            "submission_spec.md reproduce-command form; rank.py itself only "
            "reads --features, since all per-candidate data needed for ranking "
            "was already computed offline by precompute.py — see README.md)."
        ),
    )
    parser.add_argument(
        "--features", default=os.path.join("features", "features.parquet"),
        help="Path to features.parquet produced by precompute.py",
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== rank.py starting ===")
    if args.candidates:
        log.info("Candidates (unused — ranking runs entirely off precomputed features): %s", args.candidates)
    log.info("Features: %s", args.features)

    # -----------------------------------------------------------------------
    # 1. Load features
    # -----------------------------------------------------------------------
    df = pd.read_parquet(args.features, engine="pyarrow")
    log.info("Loaded %d rows × %d cols (%.2fs)", len(df), len(df.columns), time.time() - t0)

    # -----------------------------------------------------------------------
    # 2. Hard filter
    # -----------------------------------------------------------------------
    n_before = len(df)
    eligible  = df[~df["hard_out"].values & ~df["honeypot"].values].copy()
    eligible.reset_index(drop=True, inplace=True)
    log.info(
        "Eligible after hard filter: %d (removed %d hard_out/honeypot)",
        len(eligible), n_before - len(eligible),
    )

    if len(eligible) < TOP_N:
        log.error("Only %d eligible candidates — need %d. Aborting.", len(eligible), TOP_N)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3–4. Sub-scores + multipliers
    # -----------------------------------------------------------------------
    eligible = _compute_subscores(eligible)

    # -----------------------------------------------------------------------
    # 5. Primary composite
    # -----------------------------------------------------------------------
    eligible["composite"] = _composite(eligible, BASE_WEIGHTS)
    log.info(
        "Composite: min=%.4f  max=%.4f  mean=%.4f  p90=%.4f",
        eligible["composite"].min(),
        eligible["composite"].max(),
        eligible["composite"].mean(),
        eligible["composite"].quantile(0.90),
    )

    # -----------------------------------------------------------------------
    # 6. Sort: composite desc; tiebreak candidate_id asc (validator rule)
    # -----------------------------------------------------------------------
    eligible.sort_values(
        ["composite", "candidate_id"],
        ascending=[False, True],
        inplace=True,
        kind="mergesort",   # stable sort preserves secondary ordering
    )
    eligible.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------------------------
    # 7. Stability check
    # -----------------------------------------------------------------------
    primary_top_ids = set(eligible.head(STABILITY_TOP_K)["candidate_id"])
    _stability_check(eligible, primary_top_ids)

    # -----------------------------------------------------------------------
    # 8. Take top-N and generate reasoning
    # -----------------------------------------------------------------------
    top_df = eligible.head(TOP_N).copy()
    top_df["rank"] = range(1, TOP_N + 1)

    top_rows = top_df.to_dict(orient="records")
    for row in top_rows:
        row["reasoning"] = generate_reasoning(row, row["rank"])

    # -----------------------------------------------------------------------
    # 9. Write CSV
    # -----------------------------------------------------------------------
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in top_rows:
            writer.writerow([
                row["candidate_id"],
                int(row["rank"]),
                f"{row['composite']:.6f}",
                row["reasoning"],
            ])

    elapsed = time.time() - t0
    log.info("Wrote %d rows to %s (%.2fs total)", TOP_N, args.out, elapsed)
    log.info("Top-5: %s", [
        (r["candidate_id"], f"{r['composite']:.4f}")
        for r in top_rows[:5]
    ])
    log.info("=== rank.py done ===")


if __name__ == "__main__":
    main()

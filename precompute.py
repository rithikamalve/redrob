#!/usr/bin/env python3
"""
precompute.py — Offline feature computation for all candidates.

No time limit; may use the network.  Run once before evaluation.
Output (features/features.parquet) is loaded by rank.py, which must
complete within 5 minutes on CPU with no network access.

Usage
-----
    python precompute.py --candidates candidates.jsonl --out features/features.parquet

Pipeline
--------
1. Stream candidates → normalise (ingest) → run rule-based scorers per record.
2. Embed all texts with BAAI/bge-large-en-v1.5 (bi-encoder).
3. Cross-encode top RERANK_TOP_K candidates with BAAI/bge-reranker-base.
4. Write flat features.parquet (one row per candidate, ~50 columns).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Use all available CPU cores for PyTorch inference
torch.set_num_threads(os.cpu_count() or 4)

sys.path.insert(0, os.path.dirname(__file__))

from ingest import normalise_candidate, stream_candidates
from scorer.behavioral import extract_behavioral_features
from scorer.career import experience_curve, extract_career_features
from scorer.honeypot import compute_honeypot
from scorer.logistics import compute_logistics_scores
from config import RERANK_TOP_K
from scorer.semantic import (
    build_candidate_text,
    compute_semantic_scores,
    embed_candidates,
    embed_queries,
    load_model,
    load_reranker,
    rerank_candidates,
)
from scorer.skills import (
    education_tier_bonus,
    score_skills,
    skill_assessment_bonus,
    tech_keyword_score,
    weighted_skill_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute")


def _process_candidate(
    c: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Run all rule-based scorers for one normalised candidate.

    Returns (feature_row, candidate_text_or_None).
    The semantic score column is filled in by the caller after batch embedding.
    """
    profile    = c.get("profile") or {}
    signals    = c.get("redrob_signals") or {}
    career_h   = c.get("career_history") or []
    education  = c.get("education") or []
    skills_raw = c.get("skills") or []

    hp   = compute_honeypot(c)
    logi = compute_logistics_scores(c)

    yoe            = float(profile.get("years_of_experience") or 0)
    cf             = extract_career_features(career_h, yoe)
    exp_curve_sc   = experience_curve(yoe)

    group_scores, stuffer_flag, stuffer_mult = score_skills(
        skills_raw, cf["avg_desc_words"]
    )
    ws = weighted_skill_score(group_scores)
    ab = skill_assessment_bonus(signals.get("skill_assessment_scores") or {})
    eb = education_tier_bonus(education)
    tk = tech_keyword_score(career_h)

    beh = extract_behavioral_features(signals, c["_days_inactive"])

    # Reasoning helpers — not used in scoring, stored for rank.py
    headline = (profile.get("headline") or "").strip()
    current_title = headline.split("|")[0].strip() if headline else ""
    if not current_title and career_h:
        current_title = (career_h[0].get("title") or "").strip()
    current_company = (career_h[0].get("company") or "").strip() if career_h else ""
    recruiter_response_rate = float(signals.get("recruiter_response_rate") or 0)

    row: Dict[str, Any] = {
        # Identity
        "candidate_id": c["candidate_id"],

        # Honeypot
        "honeypot":        hp["honeypot"],
        "honeypot_soft":   hp["honeypot_soft"],
        "hp_signal_count": hp["signal_count"],

        # Logistics / hard gates
        "location_class":        logi["location_class"],
        "location_score":        logi["location_score"],
        "consulting_frac":       logi["consulting_frac"],
        "consulting_multiplier": logi["consulting_multiplier"],
        "domain_mismatch":       logi["domain_mismatch"],
        "hard_out":              logi["hard_out"],
        "salary_score":          logi["salary_score"],
        "workmode_score":        logi["workmode_score"],

        # Career
        "yoe":                   yoe,
        "exp_curve_score":       exp_curve_sc,
        "ml_months":             cf["ml_months"],
        "product_fraction":      cf["product_fraction"],
        "title_hop":             cf["title_hop"],
        "coding_gap":            cf["coding_gap"],
        "coding_gap_multiplier": cf["coding_gap_multiplier"],
        "founding_team_exp":     cf["founding_team_exp"],
        "recent_ml":             cf["recent_ml"],
        "avg_desc_words":        cf["avg_desc_words"],

        # Skills
        "skill_retrieval":        group_scores.get("retrieval",    0.0),
        "skill_vector_db":        group_scores.get("vector_db",    0.0),
        "skill_embeddings":       group_scores.get("embeddings",   0.0),
        "skill_eval_ranking":     group_scores.get("eval_ranking", 0.0),
        "skill_python_mlops":     group_scores.get("python_mlops", 0.0),
        "skill_llm_ops":          group_scores.get("llm_ops",      0.0),
        "weighted_skill_score":   ws,
        "stuffer_flag":           stuffer_flag,
        "stuffer_multiplier":     stuffer_mult,
        "skill_assessment_bonus": ab,
        "education_tier_bonus":   eb,
        "tech_keyword_score":     tk,

        # Behavioral
        "recency_score":       beh["recency_score"],
        "response_score":      beh["response_score"],
        "response_time_score": beh["response_time_score"],
        "github_score":        beh["github_score"],
        "notice_score":        beh["notice_score"],
        "saved_score":         beh["saved_score"],
        "interview_score":     beh["interview_score"],
        "offer_score":         beh["offer_score"],
        "open_bonus":          beh["open_bonus"],

        # Reasoning helpers
        "current_title":           current_title,
        "current_company":         current_company,
        "recruiter_response_rate": recruiter_response_rate,

        # Semantic scores — filled in after batch embedding / reranking
        "semantic_score": 0.0,
        "rerank_score":   0.0,   # 0.0 for candidates not in top-RERANK_TOP_K
    }

    text = build_candidate_text(c)
    return row, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute candidate features")
    parser.add_argument(
        "--candidates", default="candidates.jsonl",
        help="Path to candidates JSONL file (plain or .gz)",
    )
    parser.add_argument(
        "--out", default=os.path.join("features", "features.parquet"),
        help="Output path for features.parquet",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    log.info("=== precompute.py starting ===")
    log.info("Input : %s", args.candidates)
    log.info("Output: %s", args.out)

    # -----------------------------------------------------------------------
    # Phase A — rule-based scoring (streaming, per-candidate)
    # -----------------------------------------------------------------------
    rows:  List[Dict[str, Any]] = []
    texts: List[Optional[str]]  = []   # parallel to rows; None if text too sparse
    n_total = n_error = 0

    for raw in stream_candidates(args.candidates):
        n_total += 1
        try:
            normalise_candidate(raw)
            row, text = _process_candidate(raw)
            rows.append(row)
            texts.append(text)
        except Exception:
            n_error += 1
            log.exception("Failed to process %s", raw.get("candidate_id", "?"))

        if n_total % 10_000 == 0:
            elapsed = time.time() - t0
            log.info("  %6d processed | %d errors | %.1fs elapsed", n_total, n_error, elapsed)

    log.info(
        "Rule-based scoring complete: %d candidates, %d errors (%.1fs)",
        n_total, n_error, time.time() - t0,
    )

    if not rows:
        log.error("No candidates processed — aborting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase B — bi-encoder semantic embedding
    # -----------------------------------------------------------------------
    model  = load_model()
    q_vecs = embed_queries(model)
    log.info("Query embedding shape: %s", q_vecs.shape)

    valid_indices = [i for i, t in enumerate(texts) if t is not None]
    valid_texts   = [texts[i] for i in valid_indices]  # type: ignore[index]

    if valid_texts:
        c_vecs     = embed_candidates(valid_texts, model)
        sem_scores = compute_semantic_scores(c_vecs, q_vecs)
        log.info(
            "Bi-encoder scores — min=%.4f  max=%.4f  mean=%.4f  (%d valid, %d sparse)",
            sem_scores.min(), sem_scores.max(), sem_scores.mean(),
            len(valid_texts), len(rows) - len(valid_texts),
        )
        for idx, score in zip(valid_indices, sem_scores.tolist()):
            rows[idx]["semantic_score"] = float(score)
    else:
        log.warning("No valid candidate texts; semantic_score=0.0 for all.")

    # -----------------------------------------------------------------------
    # Phase C — cross-encoder reranking (top RERANK_TOP_K eligible candidates)
    # -----------------------------------------------------------------------
    # Select candidates eligible for reranking (not hard_out, not honeypot)
    # Rank them by a rough proxy score to pick the most promising RERANK_TOP_K.
    eligible_for_rerank = [
        (i, rows[i])
        for i in range(len(rows))
        if not rows[i]["hard_out"] and not rows[i]["honeypot"] and texts[i] is not None
    ]
    log.info("Eligible for reranking: %d candidates", len(eligible_for_rerank))

    if eligible_for_rerank:
        # Rough proxy: bi-encoder + ml_months fraction + recency
        def _rough_score(r: Dict[str, Any]) -> float:
            return (
                float(r["semantic_score"]) * 0.5 +
                min(float(r["ml_months"]) / 48.0, 1.0) * 0.3 +
                float(r["recency_score"]) * 0.2
            )

        eligible_for_rerank.sort(key=lambda x: _rough_score(x[1]), reverse=True)
        rerank_pool = eligible_for_rerank[:RERANK_TOP_K]

        rerank_global_indices = [i for i, _ in rerank_pool]
        rerank_texts          = [texts[i] for i in rerank_global_indices]  # type: ignore

        reranker     = load_reranker()
        rerank_scores = rerank_candidates(rerank_texts, reranker)
        log.info(
            "Rerank scores — min=%.4f  max=%.4f  mean=%.4f  (n=%d)",
            rerank_scores.min(), rerank_scores.max(), rerank_scores.mean(),
            len(rerank_scores),
        )

        for global_idx, score in zip(rerank_global_indices, rerank_scores.tolist()):
            rows[global_idx]["rerank_score"] = float(score)

    # -----------------------------------------------------------------------
    # Phase D — write parquet
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False, engine="pyarrow")

    elapsed = time.time() - t0
    log.info("Wrote %d rows × %d columns to %s (%.1fs total)", len(df), len(df.columns), args.out, elapsed)

    # Summary stats
    hard_outs  = int(df["hard_out"].sum())
    honeypots  = int(df["honeypot"].sum())
    hp_softs   = int(df["honeypot_soft"].sum())
    stuffers   = int(df["stuffer_flag"].sum())
    log.info(
        "Summary: hard_out=%d | honeypot=%d | honeypot_soft=%d | stuffer=%d",
        hard_outs, honeypots, hp_softs, stuffers,
    )
    log.info(
        "Semantic score percentiles — p25=%.3f p50=%.3f p75=%.3f p90=%.3f",
        df["semantic_score"].quantile(0.25),
        df["semantic_score"].quantile(0.50),
        df["semantic_score"].quantile(0.75),
        df["semantic_score"].quantile(0.90),
    )
    log.info("=== precompute.py done ===")


if __name__ == "__main__":
    main()

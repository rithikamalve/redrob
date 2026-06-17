"""
scorer/semantic.py — Bi-encoder embedding pipeline + cross-encoder reranking.

Bi-encoder  : BAAI/bge-large-en-v1.5 (1024-dim, strong retrieval quality).
              BGE retrieval models require an instruction prefix on queries only.
Cross-encoder: BAAI/bge-reranker-base — compares full JD text against each
              candidate text directly; much higher precision than bi-encoder
              cosine similarity. Run in precompute.py on top-2000 candidates.

Semantic score blend:
    bi_encoder_score  = 0.5×max_sim + 0.5×mean_sim  (over 5 JD queries)
    final_semantic    = 0.65×rerank_score + 0.35×bi_encoder_score  (reranked)
                      = bi_encoder_score                             (others)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    BGE_QUERY_INSTRUCTION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    JD_QUERIES,
    MIN_CANDIDATE_TEXT_WORDS,
    MODEL_CACHE_DIR,
    RERANKER_MODEL,
    RERANK_JD_TEXT,
    SEMANTIC_MAX_WEIGHT,
    SEMANTIC_MEAN_WEIGHT,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate text construction
# ---------------------------------------------------------------------------

def build_candidate_text(candidate: Dict[str, Any]) -> Optional[str]:
    """
    Construct the text string to embed for a single candidate.

    Order: headline → summary → skills (name + proficiency) →
           career roles most-recent-first (title + company + description).

    Most-recent-first matters because both the bi-encoder (512-token limit
    per text) and especially the cross-encoder (512-token limit shared
    across the JD+candidate PAIR) truncate from the end of the text once
    it's too long — measured median candidate text is ~460 tokens and
    p90 is ~670, so truncation is the common case, not the exception.
    Putting the most recent (most relevant) role first ensures truncation
    drops old, less relevant history instead of the candidate's current job.

    Returns None if the resulting text is too sparse (< MIN_CANDIDATE_TEXT_WORDS).
    """
    parts: List[str] = []

    profile = candidate.get("profile") or {}
    headline = (profile.get("headline") or "").strip()
    summary  = (profile.get("summary")  or "").strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    for skill in (candidate.get("skills") or []):
        name = (skill.get("name") or "").strip()
        prof = (skill.get("proficiency") or "").strip()
        if name:
            parts.append(f"{name} ({prof})" if prof else name)

    # career_history is already sorted most-recent-first by ingest.normalise_candidate
    for role in (candidate.get("career_history") or []):
        title   = (role.get("title")       or "").strip()
        company = (role.get("company")     or "").strip()
        desc    = (role.get("description") or "").strip()
        role_text = " ".join(filter(None, [title, company, desc]))
        if role_text:
            parts.append(role_text)

    text = " ".join(parts)
    if len(text.split()) < MIN_CANDIDATE_TEXT_WORDS:
        return None
    return text


# ---------------------------------------------------------------------------
# Bi-encoder (BGE large)
# ---------------------------------------------------------------------------

def load_model() -> "SentenceTransformer":
    """
    Load BAAI/bge-large-en-v1.5. Downloaded once to MODEL_CACHE_DIR.
    No network needed after first run.
    """
    import os
    from sentence_transformers import SentenceTransformer
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    log.info("Loading bi-encoder: %s  (cache: %s)", EMBEDDING_MODEL, MODEL_CACHE_DIR)
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)


def embed_queries(model: "SentenceTransformer") -> "np.ndarray":
    """
    Embed the 5 JD queries with the BGE instruction prefix (retrieval mode).

    Returns (5, D) L2-normalised float32.
    """
    import numpy as np
    queries_with_instruction = [
        BGE_QUERY_INSTRUCTION + q for q in JD_QUERIES
    ]
    vecs = model.encode(
        queries_with_instruction,
        batch_size=len(queries_with_instruction),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_candidates(
    texts: List[str],
    model: "SentenceTransformer",
) -> "np.ndarray":
    """
    Batch-encode candidate texts. No instruction prefix for documents.

    Returns (N, D) L2-normalised float32.
    """
    import numpy as np
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)

    log.info("Encoding %d candidates (batch=%d)...", len(texts), EMBEDDING_BATCH_SIZE)
    vecs = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def compute_semantic_scores(
    candidate_embeddings: "np.ndarray",
    query_embeddings: "np.ndarray",
) -> "np.ndarray":
    """
    0.5×max_sim + 0.5×mean_sim over 5 JD queries. Both arrays must be unit-normed.

    Returns (N,) float32 in [0.0, 1.0].
    """
    import numpy as np
    if candidate_embeddings.shape[0] == 0:
        return np.array([], dtype=np.float32)

    sims     = candidate_embeddings @ query_embeddings.T   # (N, Q) dot = cosine
    max_sim  = sims.max(axis=1)
    mean_sim = sims.mean(axis=1)
    scores   = SEMANTIC_MAX_WEIGHT * max_sim + SEMANTIC_MEAN_WEIGHT * mean_sim
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-encoder reranker (BAAI/bge-reranker-base)
# ---------------------------------------------------------------------------

def load_reranker() -> "CrossEncoder":
    """
    Load BAAI/bge-reranker-base. Downloaded once to MODEL_CACHE_DIR.
    """
    import os
    from sentence_transformers import CrossEncoder
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    log.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
    return CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        cache_folder=MODEL_CACHE_DIR,
    )


def rerank_candidates(
    candidate_texts: List[str],
    reranker: "CrossEncoder",
    batch_size: int = 32,
) -> "np.ndarray":
    """
    Score each candidate text against the full JD using the cross-encoder.

    Scores are min-max normalised to [0, 1] within this batch so they are
    on the same scale as bi-encoder scores.

    Parameters
    ----------
    candidate_texts : list of non-None candidate text strings
    reranker        : loaded CrossEncoder
    batch_size      : inference batch size (keep low for CPU)

    Returns
    -------
    np.ndarray of shape (N,), values in [0.0, 1.0], float32.
    """
    import numpy as np
    if not candidate_texts:
        return np.array([], dtype=np.float32)

    pairs = [(RERANK_JD_TEXT, text) for text in candidate_texts]
    log.info("Cross-encoding %d candidates (batch=%d)...", len(pairs), batch_size)

    raw_scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    raw_scores = np.array(raw_scores, dtype=np.float32)

    # Sigmoid → [0,1] (preserves relative order and spread better than min-max
    # when some candidates are very clearly irrelevant)
    scores = 1.0 / (1.0 + np.exp(-raw_scores))
    return scores.astype(np.float32)

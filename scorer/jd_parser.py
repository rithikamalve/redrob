"""
scorer/jd_parser.py — LLM-based JD requirement extraction (precompute.py only).

Runs ONCE per precompute.py invocation (parses a ~600-word JD, not per-candidate),
not in rank.py — fully compliant with the spec's "no hosted LLM APIs during the
ranking step" constraint, since this never touches rank.py.

Calls Groq's chat completions API to extract structured requirements from the
real JD text (config.FULL_JD_TEXT_RAW), then builds the same shape of output
(5 bi-encoder queries + 1 cross-encoder text) that the hand-written
config.JD_QUERIES / config.RERANK_JD_TEXT already provide. If GROQ_API_KEY is
unset, the API call fails, or the response can't be parsed as valid JSON
matching the expected schema, this falls back to the hand-written defaults —
precompute.py never hard-fails for lack of an API key or a flaky network call.

Result is cached to config.JD_PARSE_CACHE_PATH so repeated precompute.py runs
don't re-call the API (and so a run without network can reuse a prior result).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from config import (
    FULL_JD_TEXT_RAW,
    GROQ_API_URL,
    GROQ_MODEL,
    JD_PARSE_CACHE_PATH,
    JD_QUERIES as JD_QUERIES_FALLBACK,
    RERANK_JD_TEXT as RERANK_JD_TEXT_FALLBACK,
)

log = logging.getLogger(__name__)

_EXTRACTION_SCHEMA_PROMPT = """\
You are extracting structured hiring requirements from a job description for \
a candidate-ranking system. This feeds two models: (1) a bi-encoder that embeds \
each item independently — it needs real semantic content, not bare keywords, to \
produce a useful embedding; (2) a cross-encoder with a 512-token limit SHARED \
between this JD text and each candidate's resume combined — so the combined \
output must stay moderate in length, but each individual item should still be a \
real, specific, information-dense phrase (10-15 words), not a 3-4 word keyword \
fragment and not a full explanatory sentence either.

Good must_have_requirements item (specific, ~12 words, names real tools):
  "production embeddings-based retrieval deployed to real users — sentence-transformers, BGE, or OpenAI embeddings"
Bad (too sparse, loses the JD's actual content):
  "embeddings-based retrieval systems"
Bad (too verbose, written as a full sentence):
  "Candidates should have demonstrated production experience deploying embeddings-based retrieval systems to real users at scale"

Read the JD below and return ONLY a JSON object (no markdown, no commentary) with \
exactly these keys:

{
  "job_title": "the exact job title and company name as written in the JD, e.g. 'Senior AI Engineer, Redrob AI'",
  "role_mandate": "1 sentence, ~20-25 words, on what the role actually does",
  "must_have_requirements": ["exactly 4 items, each ~10-15 words, name specific tools/technologies from the JD"],
  "ideal_candidate_profile": "1 sentence, ~20-25 words, describing the ideal background",
  "nice_to_have": ["exactly 4 items, each ~8-12 words"],
  "hard_disqualifiers": ["exactly 4 items, each ~10-15 words, specific not generic"]
}

Be specific and concrete — pull actual phrases and technical terms from the JD \
rather than generic paraphrasing. Do not invent requirements not in the JD.

JOB DESCRIPTION:
"""

# Hard safety net: even if the LLM ignores the length guidance above, the
# final built rerank text is truncated to this many words before use.
# Calibrated against measured tokenizer behavior (~1.8-2.0 tokens/word for
# this kind of text): 170 words lands around 300-320 tokens, matching the
# hand-written fallback's 319 tokens — comfortably under the cross-encoder's
# shared 512-token pair budget, with most of it left for the candidate's text,
# while still carrying enough real content for the model to discriminate
# between candidates (an earlier, terser version collapsed nearly all
# rerank_scores to ~0.50 — see SOLUTION.md for the full story).
_RERANK_TEXT_MAX_WORDS = 170


def _call_groq(jd_text: str, api_key: str, timeout_s: float = 30.0) -> Optional[Dict[str, Any]]:
    import httpx

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": _EXTRACTION_SCHEMA_PROMPT + jd_text},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = httpx.post(GROQ_API_URL, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception:
        log.exception("Groq JD parsing call failed")
        return None

    required_keys = {
        "job_title", "role_mandate", "must_have_requirements", "ideal_candidate_profile",
        "nice_to_have", "hard_disqualifiers",
    }
    if not required_keys.issubset(parsed.keys()):
        log.warning("Groq response missing expected keys: got %s", list(parsed.keys()))
        return None

    return parsed


def _build_queries(parsed: Dict[str, Any]) -> List[str]:
    """Build 5 bi-encoder query strings from the LLM-extracted structure."""
    must_haves = parsed["must_have_requirements"]
    queries = [str(m) for m in must_haves[:4]]
    queries.append(str(parsed["ideal_candidate_profile"]))
    if parsed.get("nice_to_have"):
        queries.append("Nice to have: " + "; ".join(str(n) for n in parsed["nice_to_have"]))
    return queries[:5] if len(queries) >= 5 else (queries + JD_QUERIES_FALLBACK)[:5]


def _build_rerank_text(parsed: Dict[str, Any]) -> str:
    """Build the cross-encoder JD text — length-budgeted (see _RERANK_TEXT_MAX_WORDS)
    but deliberately NOT ultra-terse: an earlier version that squeezed items down
    to 3-4 word keyword fragments collapsed rerank_score to ~0.50 for nearly all
    candidates (not enough real content for the cross-encoder to discriminate on).

    Empirically verified (direct A/B test on the same 15 candidates, same
    reranker, only the JD text varied) that leading with the explicit
    job_title ("Senior AI Engineer, Redrob AI") rather than going straight
    into a paraphrased role_mandate matters a lot: without it, scores
    collapsed to ~0.500-0.501 across nearly all candidates even with
    otherwise rich, specific must-have/disqualifier content; with the title
    leading, scores spread to ~0.50-0.54. Hypothesis: the cross-encoder
    needs an explicit, concrete anchor for "what is this text" before it can
    usefully compare paraphrased requirement text against a resume — bge-
    reranker-base wasn't trained to start cold on an abstract paraphrase.
    Not fully understood, but reproduced 3 times, so treated as a hard
    requirement on the text's structure, not just a one-off observation.
    """
    parts = [str(parsed["job_title"]) + "."]
    parts.append(str(parsed["role_mandate"]))
    parts.append("Must have: " + "; ".join(str(m) for m in parsed["must_have_requirements"][:4]) + ".")
    if parsed.get("hard_disqualifiers"):
        parts.append("Reject: " + "; ".join(str(d) for d in parsed["hard_disqualifiers"][:4]) + ".")
    if parsed.get("nice_to_have"):
        parts.append("Nice to have: " + "; ".join(str(n) for n in parsed["nice_to_have"][:3]) + ".")
    text = "\n".join(parts)

    # Note: the LLM occasionally drops a space after punctuation within an
    # item (e.g. "strongPython", "XGBoost-basedor"). A naive regex fix for
    # this ("insert space before a capital letter after a lowercase letter")
    # was tried and reverted — it corrupts legitimate technical terms that
    # are exactly the signal this text needs to preserve, e.g. "LoRA" ->
    # "Lo RA", "QLoRA" -> "QLo RA". Left as a rare cosmetic glitch instead;
    # it doesn't affect tokenization correctness, just readability.

    words = text.split()
    if len(words) > _RERANK_TEXT_MAX_WORDS:
        log.warning(
            "LLM-built rerank text was %d words, truncating to %d (prompt's "
            "brevity instructions weren't followed strictly).",
            len(words), _RERANK_TEXT_MAX_WORDS,
        )
        text = " ".join(words[:_RERANK_TEXT_MAX_WORDS])
    return text


def get_jd_queries_and_rerank_text() -> Tuple[List[str], str, bool]:
    """
    Returns (jd_queries, rerank_jd_text, used_llm).

    Tries, in order: disk cache -> live Groq call -> hand-written fallback.
    Never raises — always returns something usable.
    """
    if os.path.exists(JD_PARSE_CACHE_PATH):
        try:
            with open(JD_PARSE_CACHE_PATH, encoding="utf-8") as f:
                cached = json.load(f)
            log.info("Using cached LLM-parsed JD requirements: %s", JD_PARSE_CACHE_PATH)
            return cached["jd_queries"], cached["rerank_jd_text"], True
        except Exception:
            log.warning("JD parse cache exists but is unreadable; ignoring it", exc_info=True)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.warning(
            "GROQ_API_KEY not set — using hand-written JD_QUERIES/RERANK_JD_TEXT "
            "from config.py instead of LLM-based JD parsing."
        )
        return JD_QUERIES_FALLBACK, RERANK_JD_TEXT_FALLBACK, False

    log.info("Calling Groq (%s) to parse JD into structured requirements...", GROQ_MODEL)
    parsed = _call_groq(FULL_JD_TEXT_RAW, api_key)
    if parsed is None:
        log.warning("Groq JD parsing failed — falling back to hand-written JD_QUERIES/RERANK_JD_TEXT.")
        return JD_QUERIES_FALLBACK, RERANK_JD_TEXT_FALLBACK, False

    jd_queries = _build_queries(parsed)
    rerank_text = _build_rerank_text(parsed)

    try:
        os.makedirs(os.path.dirname(JD_PARSE_CACHE_PATH), exist_ok=True)
        with open(JD_PARSE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"raw_extraction": parsed, "jd_queries": jd_queries, "rerank_jd_text": rerank_text},
                f, indent=2,
            )
        log.info("Cached LLM-parsed JD requirements to %s", JD_PARSE_CACHE_PATH)
    except Exception:
        log.warning("Failed to write JD parse cache (non-fatal)", exc_info=True)

    return jd_queries, rerank_text, True

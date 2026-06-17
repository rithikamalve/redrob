"""
config.py — Single source of truth for every constant, weight, threshold,
keyword list, and JD query string used across the pipeline.

All other modules import from here. To tune the ranker, change values here only.
"""

from __future__ import annotations

import datetime
from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Runtime reference date — computed once at import; never hardcode a date string
# ---------------------------------------------------------------------------
DATE_TODAY: datetime.date = datetime.date.today()

# ---------------------------------------------------------------------------
# Submission output
# ---------------------------------------------------------------------------
TOP_N: int = 100
REASONING_MAX_CHARS: int = 250

# ---------------------------------------------------------------------------
# Stability check (3 weight configs, ≥15/20 overlap required in top-20)
# ---------------------------------------------------------------------------
STABILITY_MIN_OVERLAP: int = 15
STABILITY_TOP_K: int = 20

# ---------------------------------------------------------------------------
# Composite base weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
BASE_WEIGHTS: Dict[str, float] = {
    "jd_fit":     0.42,
    "career":     0.32,
    "behavioral": 0.21,
    "logistics":  0.05,
}

# Three configs for stability validation
STABILITY_WEIGHT_CONFIGS: List[Dict[str, float]] = [
    {"jd_fit": 0.42, "career": 0.32, "behavioral": 0.21, "logistics": 0.05},  # primary
    {"jd_fit": 0.35, "career": 0.38, "behavioral": 0.22, "logistics": 0.05},  # career-heavy
    {"jd_fit": 0.50, "career": 0.25, "behavioral": 0.20, "logistics": 0.05},  # fit-heavy
]

# ---------------------------------------------------------------------------
# JD-fit sub-weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
JD_FIT_WEIGHTS: Dict[str, float] = {
    "semantic":     0.55,
    "skills":       0.25,
    "tech_keyword": 0.20,
}

# ---------------------------------------------------------------------------
# Career sub-weights (main components only; must sum to 1.0)
# Bonuses (startup, recent_ml, education) are additive before clip(0, 1)
# ---------------------------------------------------------------------------
CAREER_WEIGHTS: Dict[str, float] = {
    "ml_component":    0.35,
    "product_fraction": 0.30,
    "trajectory":      0.25,
    "exp_curve":       0.10,
}
STARTUP_BONUS_VALUE: float = 0.05
RECENT_ML_BONUS_VALUE: float = 0.05

# ---------------------------------------------------------------------------
# Behavioral sub-weights (must sum to 1.0; open_to_work bonus is additive)
# ---------------------------------------------------------------------------
BEHAVIORAL_WEIGHTS: Dict[str, float] = {
    "recency":       0.28,
    "response":      0.18,
    "response_time": 0.14,
    "github":        0.14,
    "notice":        0.10,
    "saved":         0.08,
    "interview":     0.04,
    "offer":         0.04,
}
OPEN_TO_WORK_BONUS: float = 0.10

# ---------------------------------------------------------------------------
# Logistics sub-weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
LOGISTICS_WEIGHTS: Dict[str, float] = {
    "location": 0.60,
    "workmode": 0.25,
    "salary":   0.15,
}

# ---------------------------------------------------------------------------
# Score multipliers applied to composite after base score
# ---------------------------------------------------------------------------
HONEYPOT_SOFT_MULT: float = 0.65   # 1 honeypot signal
STUFFER_MULT: float       = 0.70   # keyword stuffer detected
CODING_GAP_MULT: float    = 0.75   # management-only recent titles

# ---------------------------------------------------------------------------
# Location classification scores
# ---------------------------------------------------------------------------
LOCATION_SCORES: Dict[str, float] = {
    "tier1_india":  1.00,
    "tier2_india":  0.80,
    "remote_india": 0.70,
    "intl_open":    0.50,
    "intl_no":      0.00,   # hard_out — removed before scoring
}

# Work-mode scores (JD says hybrid-preferred, Pune/Noida offices Tue/Thu)
WORKMODE_SCORES: Dict[str, float] = {
    "hybrid":   1.00,
    "flexible": 1.00,
    "remote":   0.75,
    "onsite":   0.60,
}

# ---------------------------------------------------------------------------
# India city tier sets (lowercase; checked via substring match)
# ---------------------------------------------------------------------------
TIER1_INDIA_CITIES: FrozenSet[str] = frozenset({
    "bengaluru", "bangalore",
    "noida", "greater noida",
    "gurugram", "gurgaon",
    "delhi", "new delhi", "ncr", "delhi ncr",
    "faridabad", "ghaziabad",          # Delhi NCR satellites
    "pune",
    "hyderabad", "secunderabad",
    "mumbai", "bombay", "navi mumbai", "thane",
    "chennai", "madras",
    "kolkata", "calcutta",
})

TIER2_INDIA_CITIES: FrozenSet[str] = frozenset({
    "ahmedabad", "jaipur", "lucknow", "chandigarh", "indore",
    "bhopal", "nagpur", "coimbatore", "kochi", "thiruvananthapuram",
    "visakhapatnam", "vizag", "surat", "vadodara", "nashik",
    "aurangabad", "mysuru", "mysore", "hubli", "mangalore",
    "bhubaneswar", "patna", "ranchi", "raipur", "dehradun",
    "guwahati", "bhubaneshwar", "trichy", "madurai", "agra",
    "kanpur", "amritsar", "jalandhar", "srinagar", "shimla",
})

# Canonical city name map — normalise before tier lookup
CITY_CANON: Dict[str, str] = {
    "bangalore":      "bengaluru",
    "new delhi":      "delhi",
    "gurgaon":        "gurugram",
    "bombay":         "mumbai",
    "madras":         "chennai",
    "calcutta":       "kolkata",
    "secunderabad":   "hyderabad",
    "greater noida":  "noida",
    "ncr":            "delhi",
    "delhi ncr":      "delhi",
    "navi mumbai":    "mumbai",
    "thane":          "mumbai",
    "mysore":         "mysuru",
    "vizag":          "visakhapatnam",
    "bhubaneshwar":   "bhubaneswar",
}

# ---------------------------------------------------------------------------
# Consulting / IT-services firms (substring match on lowercased company name)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS: FrozenSet[str] = frozenset({
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl technologies",
    "hcl tech",
    "tech mahindra",
    "mphasis",
    "ltimindtree",
    "l&t infotech",
    "larsen & toubro infotech",
    "hexaware",
    "niit technologies",
    "syntel",
    "kpit technologies",
    "birlasoft",
})

# Consulting fraction thresholds
CONSULTING_HARD_OUT_FRACTION: float = 1.00   # entire career → hard_out
CONSULTING_MULTIPLIER_THRESHOLD: float = 0.50  # above this → linear penalty

# ---------------------------------------------------------------------------
# Salary fit (INR Lakhs Per Annum)
# ---------------------------------------------------------------------------
JD_SALARY_MIN: float = 40.0
JD_SALARY_MAX: float = 80.0
SALARY_NO_OVERLAP_SCORE: float  = 0.20
SALARY_UNDISCLOSED_SCORE: float = 0.50

# ---------------------------------------------------------------------------
# Skill taxonomy — 6 groups, keyword substring lists
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "retrieval": [
        "elasticsearch", "opensearch", "solr", "bm25", "lucene",
        "hybrid search", "sparse retrieval", "inverted index",
        "keyword search", "full-text search",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma",
        "pgvector", "scann", "ann", "vector database", "vector store",
        "approximate nearest neighbor", "hnsw", "annoy",
    ],
    "embeddings": [
        "sentence-transformers", "sentence transformers", "bge", "e5",
        "openai embeddings", "dense retrieval", "semantic search",
        "bi-encoder", "cross-encoder", "bert", "embedding model",
        "text embedding", "dense passage retrieval", "dpr", "colbert",
    ],
    "llm_ops": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
        "rag", "retrieval augmented", "langchain", "llm", "gpt", "llama",
        "prompt engineering", "instruction tuning", "splade",
    ],
    "eval_ranking": [
        "ndcg", "mrr", "map", "mean average precision", "a/b test",
        "offline eval", "online eval", "learning to rank", "ltr",
        "xgboost ltr", "ranknet", "lambdamart", "evaluation framework",
        "precision at k", "recall at k", "reranking",
    ],
    "python_mlops": [
        "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
        "fastapi", "flask", "numpy", "pandas", "docker", "kubernetes",
        "mlflow", "airflow", "spark", "kafka", "ray", "celery",
    ],
}

# Skill group weights (must sum to 1.0)
SKILL_WEIGHTS: Dict[str, float] = {
    "retrieval":    0.25,
    "vector_db":    0.20,
    "embeddings":   0.20,
    "eval_ranking": 0.20,
    "python_mlops": 0.10,
    "llm_ops":      0.05,
}

# Self-reported proficiency → numeric multiplier
PROFICIENCY_WEIGHT: Dict[str, float] = {
    "expert":       1.00,
    "advanced":     0.80,
    "intermediate": 0.50,
    "beginner":     0.20,
}

# Stuffer detection thresholds
STUFFER_MAX_SKILLS: int        = 20   # skill count ABOVE this
STUFFER_MIN_DESC_WORDS: int    = 40   # avg desc word count BELOW this

# Endorsement boost cap (per skill)
ENDORSEMENT_BOOST_MAX: float   = 0.20
ENDORSEMENT_BOOST_DIVISOR: int = 20   # 20 endorsements → full cap

# Duration saturation (months at which duration_w = 1.0)
DURATION_SATURATION_MONTHS: int = 24

# ---------------------------------------------------------------------------
# Platform skill-assessment bonus
# ---------------------------------------------------------------------------
ASSESSMENT_BONUS_MAX: float = 0.10
ASSESSMENT_CORE_TERMS: List[str] = [
    "retrieval", "elasticsearch", "vector", "embedding", "nlp",
    "python", "machine learning", "ranking", "recommendation", "search",
    "pytorch", "deep learning", "information retrieval",
]

# ---------------------------------------------------------------------------
# Education tier bonus
# ---------------------------------------------------------------------------
EDUCATION_TIER_BONUS: Dict[str, float] = {
    "tier_1": 0.05,
    "tier_2": 0.03,
    "tier_3": 0.01,
    "tier_4": 0.00,
    "unknown": 0.00,
}

# ---------------------------------------------------------------------------
# Career feature keywords
# ---------------------------------------------------------------------------
ML_AI_TITLE_KEYWORDS: List[str] = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp", "deep learning", "research scientist", "applied scientist",
    "recommendation", "search engineer", "ranking", "retrieval",
    "computer vision", "cv engineer",   # included for ml_months; filtered by domain gate
]

MGMT_KEYWORDS: List[str] = [
    "manager", "director", "vp", "vice president",
    "head of", "principal", "architect",
]

IC_KEYWORDS: List[str] = [
    "engineer", "scientist", "developer", "analyst",
    "researcher", "specialist", "lead",
]

# Startup company sizes (for founding_team_exp)
STARTUP_SIZES: FrozenSet[str] = frozenset({"1-10", "11-50", "51-200"})

# Career thresholds
ML_MONTHS_FULL_SCORE: int             = 48    # 4 years → score = 1.0
TITLE_HOP_GRACE_DAYS: int             = 3 * 365  # first 3 yrs of career excluded
TITLE_HOP_MIN_TENURE_MONTHS: int      = 18    # avg tenure < this → title_hop
RECENT_ROLE_WINDOW_MONTHS: int        = 36    # "recent" role lookback

# Experience-curve breakpoints: (yoe_upper_bound_exclusive, score)
# Sorted ascending; if yoe exceeds all, use EXPERIENCE_CURVE_DEFAULT
EXPERIENCE_CURVE_BREAKPOINTS: List[Tuple[float, float]] = [
    (3.0,  0.50),
    (5.0,  0.75),
    (8.0,  1.00),
    (10.0, 0.90),
    (12.0, 0.80),
]
EXPERIENCE_CURVE_DEFAULT: float = 0.70

# Trajectory component scores
TRAJECTORY_CLEAN: float = 1.00   # no hop, no coding_gap
TRAJECTORY_BOTH:  float = 0.50   # hop AND coding_gap
TRAJECTORY_HOP:   float = 0.70   # hop only
TRAJECTORY_GAP:   float = 0.75   # coding_gap only

# ---------------------------------------------------------------------------
# Domain mismatch keywords (Gate C)
# ---------------------------------------------------------------------------
CV_SPEECH_KEYWORDS: List[str] = [
    "computer vision", "image classification", "object detection",
    "image segmentation", "yolo", "opencv", "resnet",
    "speech recognition", "tts", "text-to-speech", "asr",
    "automatic speech recognition", "robotics", "lidar", "slam",
]

NLP_IR_KEYWORDS: List[str] = [
    "nlp", "natural language", "retrieval", "ranking", "recommendation",
    "search", "embedding", "language model", "text", "information retrieval",
    "question answering", "summarization", "sentiment", "ner",
    "named entity", "text classification", "semantic",
]

# Implausible seniority (honeypot Signal 4)
SENIORITY_TITLE_KEYWORDS: List[str] = [
    "vp", "vice president", "director", "cto", "ceo", "coo",
    "chief", "c-suite",
]
SENIORITY_MAX_YOE: float = 4.0

# ---------------------------------------------------------------------------
# Technical keyword overlay (career description scan)
# ---------------------------------------------------------------------------
TECHNICAL_KEYWORDS: List[str] = [
    "faiss", "hnsw", "scann", "annoy", "bi-encoder", "cross-encoder",
    "dense passage retrieval", "dpr", "colbert", "splade", "bm25",
    "ndcg", "mrr", "map@", "p@10", "offline eval", "online eval",
    "vector index", "embedding drift", "index refresh", "reranking",
    "retrieval pipeline", "hybrid retrieval", "two-tower", "siamese",
    "lambdamart", "ranknet", "learning to rank", "reciprocal rank fusion",
]
TECH_KEYWORD_FULL_SCORE_HITS: int = 5  # 5 hits → score = 1.0

# ---------------------------------------------------------------------------
# Behavioral signal thresholds
# ---------------------------------------------------------------------------
RECENCY_DECAY_DAYS: float         = 90.0   # e^(-days / 90): 90d → 0.37
RESPONSE_TIME_DECAY_HOURS: float  = 72.0   # e^(-hours / 72): 3d → 0.37
NOTICE_LINEAR_MAX_DAYS: int       = 120    # 0d → 1.0, 120d → 0.0
MIN_SEARCH_FOR_RESPONSE: int      = 5      # below this → response_rate unknown

GITHUB_UNKNOWN_SCORE: float       = 0.35   # no GitHub linked (-1 sentinel)
OFFER_UNKNOWN_SCORE: float        = 0.50   # no offer history (-1 sentinel)
RESPONSE_UNKNOWN_SCORE: float     = 0.50   # insufficient search exposure

SAVED_LOG_MAX: int                = 20     # log1p(20) → saved_score = 1.0

# ---------------------------------------------------------------------------
# Honeypot detection thresholds
# ---------------------------------------------------------------------------
HONEYPOT_HARD_SIGNAL_COUNT: int    = 2      # ≥2 signals → honeypot=True
TIMELINE_CONTRADICTION_MONTHS: int = 18     # |stated - actual| > 18 → signal
EXPERT_ZERO_DUR_MIN_SKILLS: int    = 5      # ≥5 expert+0mo skills → signal
YOE_GRAD_SLACK_YEARS: int          = 3      # YOE > (current_yr - grad_yr + 3) → signal
PERFECT_STALE_INACTIVE_DAYS: int   = 730    # 2 years inactive with 100% completeness
PERFECT_COMPLETENESS: float        = 100.0

# Salary swap: description-text proxy for product-embedded consulting work
CONSULTING_PRODUCT_PROXY_WORDS: List[str] = [
    "embedded at", "client product", "product team", "worked with",
    "seconded to", "dedicated team",
]

# ---------------------------------------------------------------------------
# Semantic embedding — bi-encoder (BGE large for high-quality retrieval)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str      = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 512   # MiniLM is tiny — large batches are fine
SEMANTIC_MAX_WEIGHT: float  = 0.5
SEMANTIC_MEAN_WEIGHT: float = 0.5
# Instruction prefix for BGE models — empty for MiniLM
BGE_QUERY_INSTRUCTION: str = ""
# Model cache inside project (avoids ~/.cache pollution)
MODEL_CACHE_DIR: str = "models"

# Minimum words for a candidate text to be considered embeddable
MIN_CANDIDATE_TEXT_WORDS: int = 15

# ---------------------------------------------------------------------------
# Cross-encoder reranker (runs in precompute.py on top-K bi-encoder results)
# ---------------------------------------------------------------------------
RERANKER_MODEL: str    = "BAAI/bge-reranker-base"
RERANK_TOP_K: int      = 500    # candidates to cross-encode (keep fast)
RERANK_SCORE_WEIGHT: float = 0.65  # blend: 0.65×rerank + 0.35×bi-encoder

# Full JD text used as the query for cross-encoder comparison.
# Distilled from the actual job_description.docx — every requirement,
# disqualifier, and "explicitly do NOT want" item below is taken directly
# from that document, not paraphrased or invented. Kept deliberately short
# (~150 tokens): max_length=512 in load_reranker() applies to the JD+candidate
# PAIR combined, not the JD alone — an earlier, longer draft of this text
# (824 tokens on its own) would have left almost no room for the candidate's
# text, making every comparison meaningless. Logistics details (location,
# notice period, salary) are deliberately omitted here since they're already
# scored numerically by scorer/logistics.py and scorer/behavioral.py —
# duplicating them here would just burn token budget for no extra signal.
RERANK_JD_TEXT: str = (
    "Senior AI Engineer, Redrob AI. Owns ranking, retrieval, and matching systems for a "
    "recruiting platform: replacing BM25 + rules with embeddings, hybrid retrieval, LLM "
    "re-ranking, and rigorous evaluation (NDCG, MRR, MAP, A/B testing).\n"
    "Must have: production embeddings-based retrieval (sentence-transformers, BGE, E5) "
    "deployed to real users; production vector DB or hybrid search (Pinecone, Weaviate, "
    "Qdrant, Milvus, Elasticsearch, FAISS); strong Python; hands-on ranking evaluation "
    "framework design.\n"
    "Reject: pure research/academic background with no production deployment; AI experience "
    "that is only recent (<12mo) LangChain-to-OpenAI projects; senior engineers who haven't "
    "written production code in 18+ months (architecture/tech-lead only); career spent "
    "entirely at consulting/IT-services firms (TCS, Infosys, Wipro, Accenture, Cognizant, "
    "Capgemini) with no product-company experience; title-chasers switching companies every "
    "~1.5 years; computer vision/speech/robotics specialists without NLP/IR exposure.\n"
    "Nice to have: LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR-tech background, "
    "open-source contributions."
)

# JD decomposed into 5 targeted query strings for the bi-encoder, mirroring
# the actual structure of job_description.docx (must-haves, eval rigor,
# shipped-production-code bar, ideal-candidate narrative, nice-to-haves) —
# not a generic AI/ML buzzword list.
JD_QUERIES: List[str] = [
    # Q1 — Must-have: production embeddings retrieval + vector DB/hybrid search
    (
        "Production experience with embeddings-based retrieval systems deployed to real users — "
        "sentence-transformers, OpenAI embeddings, BGE, E5. Handled embedding drift, index "
        "refresh, retrieval-quality regression in production. Production vector databases or "
        "hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
        "Elasticsearch, FAISS."
    ),
    # Q2 — Must-have: rigorous ranking evaluation
    (
        "Hands-on experience designing evaluation frameworks for ranking systems. NDCG, MRR, "
        "MAP, offline-to-online correlation, A/B test interpretation. Systematic, rigorous "
        "measurement of ranking quality, not just intuition."
    ),
    # Q3 — Must-have: strong Python, shipped production code recently (not architecture-only)
    (
        "Strong Python, high code quality. Currently writing production code, not purely in an "
        "architecture or tech-lead role. Shipped an end-to-end ranking, search, or recommendation "
        "system to real users at meaningful scale at a product company, not a pure consulting or "
        "services firm."
    ),
    # Q4 — Ideal-candidate narrative: founding-team / startup shipper mindset
    (
        "Founding team or early-stage startup experience, product-company background, scrappy "
        "product-engineering attitude, ships a working system in a week even if imperfect, owns "
        "the full stack, comfortable with ambiguity, strong opinions on hybrid vs dense retrieval "
        "and when to fine-tune vs prompt an LLM, defensible with reference to real systems built."
    ),
    # Q5 — Nice-to-have: LLM fine-tuning, learning-to-rank, HR-tech, distributed systems
    (
        "LLM fine-tuning with LoRA, QLoRA, or PEFT. Learning-to-rank models, XGBoost-based or "
        "neural. Prior HR-tech, recruiting-tech, or marketplace product experience. Distributed "
        "systems or large-scale inference optimization. Open-source AI/ML contributions."
    ),
]

# ---------------------------------------------------------------------------
# Internal weight validation — runs at import time; catches misconfiguration
# ---------------------------------------------------------------------------
def _validate_weights() -> None:
    checks = {
        "BASE_WEIGHTS":       BASE_WEIGHTS,
        "JD_FIT_WEIGHTS":     JD_FIT_WEIGHTS,
        "CAREER_WEIGHTS":     CAREER_WEIGHTS,
        "BEHAVIORAL_WEIGHTS": BEHAVIORAL_WEIGHTS,
        "LOGISTICS_WEIGHTS":  LOGISTICS_WEIGHTS,
        "SKILL_WEIGHTS":      SKILL_WEIGHTS,
    }
    for name, weights in checks.items():
        total = round(sum(weights.values()), 10)
        assert total == 1.0, (
            f"{name} must sum to 1.0, got {total}. "
            f"Values: {weights}"
        )
    for i, cfg in enumerate(STABILITY_WEIGHT_CONFIGS):
        total = round(sum(cfg.values()), 10)
        assert total == 1.0, (
            f"STABILITY_WEIGHT_CONFIGS[{i}] must sum to 1.0, got {total}"
        )


_validate_weights()

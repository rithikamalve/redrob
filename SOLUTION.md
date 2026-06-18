# Redrob Hackathon — Intelligent Candidate Ranking: Full Solution Design

---

## 0. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         PHASE 1 — precompute.py                                ║
║                    (runs once offline, no time limit)                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  candidates.jsonl.gz (100K records, ~465 MB uncompressed)
          │
          ▼
  ┌───────────────────┐
  │   INGEST LAYER    │  gzip + json, UTF-8 norm, date clamp,
  │                   │  salary swap, career date merge
  └────────┬──────────┘
           │ 100,000 candidates
           │
           ├──────────────────────────────────────────────────────────┐
           │                                                          │
           ▼                                                          ▼
  ┌─────────────────────────┐                            ┌─────────────────────────┐
  │   HONEYPOT DETECTOR     │                            │  HARD GATE COMPUTER     │
  │                         │                            │                         │
  │  Signal 1: timeline     │                            │  Gate A: Location       │
  │    contradiction        │                            │    (India / Intl)       │
  │  Signal 2: expert+0mo   │                            │                         │
  │  Signal 3: YOE vs grad  │                            │  Gate B: Consulting     │
  │  Signal 4: VP + <4 yrs  │                            │    fraction (0→1.0x)    │
  │  Signal 5: perfect+stale│                            │                         │
  │                         │                            │  Gate C: Domain         │
  │  ≥2 signals → hard_out  │                            │    mismatch (CV/speech) │
  │  1 signal  → 0.65x mult │                            │                         │
  └────────────┬────────────┘                            └────────────┬────────────┘
               │                                                      │
               └──────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  RULE-BASED EXTRACTOR    │
                    │                          │
                    │  Career Features:        │
                    │  • ml_months             │
                    │  • product_fraction      │
                    │  • title_hop             │
                    │  • coding_gap            │
                    │  • founding_team_exp     │
                    │  • recent_ml             │
                    │  • avg_desc_words        │
                    │                          │
                    │  Skills Features:        │
                    │  • 6 taxonomy groups     │
                    │  • synonym map applied   │
                    │  • proficiency × tenure  │
                    │  • endorsement boost     │
                    │  • stuffer_flag check    │
                    │  • assessment_bonus from │
                    │    platform assessments  │
                    │                          │
                    │  Behavioral Features:    │
                    │  • recency (exp decay)   │
                    │  • response_score        │
                    │  • github_score          │
                    │  • notice_score          │
                    │  • saved_by_recruiters   │
                    │  • interview / offer     │
                    │                          │
                    │  Logistics Features:     │
                    │  • location_tier         │
                    │  • salary_fit            │
                    │  • workmode_score        │
                    │                          │
                    │  Tech Keyword Overlay:   │
                    │  • FAISS, HNSW, NDCG,   │
                    │    bi-encoder, etc.      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  JD PARSER (Groq LLM)    │
                    │                          │
                    │  Parses real JD once →   │
                    │  must-haves, reject list,│
                    │  nice-to-haves           │
                    │                          │
                    │  Builds: 5 JD_QUERIES +  │
                    │  RERANK_JD_TEXT (~300tok,│
                    │  hard word-cap enforced) │
                    │                          │
                    │  No GROQ_API_KEY, or call│
                    │  fails → falls back to   │
                    │  hand-written defaults   │
                    │  in config.py (never     │
                    │  hard-fails the run)     │
                    │                          │
                    │  Cached to disk so reruns│
                    │  don't re-call the API   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  SEMANTIC EMBEDDER       │
                    │                          │
                    │  Model: MiniLM-L6-v2     │
                    │  Dim: 384                │
                    │  Batch: 512              │
                    │                          │
                    │  5 JD Query Vectors —    │
                    │  from JD PARSER above    │
                    │  (must-haves, ideal      │
                    │  profile, nice-to-haves; │
                    │  hand-written fallback   │
                    │  has same 5-topic shape) │
                    │                          │
                    │  Candidate text =        │
                    │  headline+summary+skills │
                    │  + ALL roles, most-recent│
                    │  -first (no cap; trunc.  │
                    │  drops oldest, not newest│
                    │                          │
                    │  Score = 0.5×max_sim     │
                    │        + 0.5×mean_sim    │
                    └────────────┬─────────────┘
                                 │ semantic_score for all 100K
                                 ▼
                    ┌──────────────────────────┐
                    │  CROSS-ENCODER RERANKER  │
                    │                          │
                    │  Model: bge-reranker-base│
                    │  Pool: top 500 by proxy  │
                    │   score (not hard_out/   │
                    │   honeypot), ranked by   │
                    │   0.5×semantic_score     │
                    │ + 0.3×ml_months_frac     │
                    │ + 0.2×recency_score      │
                    │                          │
                    │  Pairs (JD_full_text,    │
                    │  candidate_text) scored  │
                    │  with full cross-        │
                    │  attn, sigmoid→[0,1]     │
                    │                          │
                    │  rerank_score = 0.0 for  │
                    │  candidates outside the  │
                    │  top-500 pool            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   features.parquet       │  ~200 MB
                    │   (one row per cand,     │
                    │   incl. rerank_score)    │
                    └──────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════════════╗
║                           PHASE 2 — rank.py                                    ║
║                    (≤5 min wall-clock, CPU, no network)                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  features.parquet
          │
          ▼
  ┌───────────────────┐
  │  LOAD + FILTER    │  Drop hard_out=True and honeypot=True
  │  ~2 seconds       │  ~35K–50K candidates remain
  └────────┬──────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────┐
  │            COMPOSITE SCORER  (vectorized numpy)     │
  │                                                     │
  │  blended_semantic = semantic_score, OR for the       │
  │    500 reranked candidates:                          │
  │    0.65×rerank_score + 0.35×semantic_score           │
  │                                                     │
  │  jd_fit     = 0.55 × blended_semantic                │
  │             + 0.25 × weighted_skill_group_score      │
  │             + 0.20 × tech_keyword_score              │
  │                                                     │
  │  career     = 0.35 × ml_component                   │
  │             + 0.30 × product_fraction                │
  │             + 0.25 × trajectory_component            │
  │             + 0.10 × experience_curve                │
  │             + 0.05 × startup_bonus                   │
  │             + 0.05 × recent_ml_bonus                 │
  │             + 0.00–0.05 × education_tier_bonus       │
  │                                                     │
  │  behavioral = 0.28 × recency_score                  │
  │             + 0.18 × response_score                  │
  │             + 0.14 × response_time_score             │
  │             + 0.14 × github_score                    │
  │             + 0.10 × notice_score                    │
  │             + 0.08 × saved_score                     │
  │             + 0.04 × interview_score                 │
  │             + 0.04 × offer_score                     │
  │             + open_bonus (0.10 if open_to_work_flag) │
  │                                                     │
  │  logistics  = 0.60 × location_score                 │
  │             + 0.25 × workmode_score                  │
  │             + 0.15 × salary_score                    │
  │                                                     │
  │  base       = 0.42×jd_fit + 0.32×career             │
  │             + 0.21×behavioral + 0.05×logistics       │
  │                                                     │
  │  composite  = base × consulting_mult                 │
  │                    × stuffer_mult                    │
  │                    × coding_gap_mult                 │
  │                    × honeypot_soft_mult              │
  │                    × diversity_mult                  │
  │                    .clip(0, 1)                       │
  │                                                     │
  │  diversity_mult: within any (company, title)         │
  │  cluster, only top-2 by jd_fit_score keep 1.0x;       │
  │  rest get 0.8x — keeps synthetic filler clusters      │
  │  (e.g. 485 "Pied Piper / Accountant" candidates)       │
  │  from crowding out genuine diversity                  │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  SORT + TIEBREAK       │
              │                        │
              │  1. composite DESC     │
              │  2. candidate_id ASC   │
              │     (validator enforced│
              │      — lines 139-144) │
              │  Take top 100          │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  STABILITY CHECK       │
              │                        │
              │  Run configs B & C     │
              │  Assert ≥15/20 overlap │
              │  in top-20 across all  │
              │  3 weight configs      │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  REASONING GENERATOR   │
              │                        │
              │  Deterministic,        │
              │  fact-grounded,        │
              │  no LLM, no hallu-     │
              │  cination possible     │
              │  (only reads actual    │
              │  profile fields)       │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  OUTPUT VALIDATOR      │
              │                        │
              │  Assert: 100 rows      │
              │  Assert: ranks 1-100   │
              │  Assert: IDs in pool   │
              │  Assert: scores ↓      │
              │  Assert: 4 cols only   │
              │  Write UTF-8 CSV       │
              └────────────────────────┘
                           │
                           ▼
                    team_xxx.csv
         (candidate_id, rank, score, reasoning)
```

---

## 00. Tech Stack

| Layer | Library / Tool | Version | Why This Choice |
|---|---|---|---|
| **Runtime** | Python | 3.11+ | f-strings, `tomllib`, faster CPython |
| **Data loading** | `gzip` + `json` | stdlib | No dependency; line-by-line streaming avoids loading 465 MB at once |
| **Feature table** | `pandas` | 2.x | Vectorized scoring across 100K rows; `.parquet` I/O via pyarrow |
| **Parquet I/O** | `pyarrow` | 14.x | Fast columnar read/write; ~10× smaller than CSV for feature store |
| **Numeric ops** | `numpy` | 1.26+ | Vectorized composite scoring, exp decay, clip — all in-memory, sub-second |
| **Embeddings** | `sentence-transformers` | 2.x | Wraps MiniLM cleanly; handles batching, tokenization, normalization |
| **Embedding model (bi-encoder)** | `all-MiniLM-L6-v2` | — | 22 MB, 384-dim, CPU-optimized, 512-token limit; fast enough to embed all 100K candidates as a first-pass retriever |
| **Reranker (cross-encoder)** | `BAAI/bge-reranker-base` via `sentence-transformers.CrossEncoder` | — | Full cross-attention between JD text and candidate text — far more precise than cosine similarity, but too slow to run on all 100K; applied only to the top 500 candidates surfaced by the bi-encoder (classic two-stage retrieval architecture) |
| **ML framework** | `torch` (CPU) | 2.x | Required by sentence-transformers; CPU-only build keeps install lean; `torch.set_num_threads(os.cpu_count())` set in `precompute.py` to use all cores |
| **Date parsing** | `python-dateutil` | 2.x | Handles all date string formats in the dataset without manual strptime |
| **Language detect** | `langdetect` | 1.x | Skip non-English career descriptions before embedding |
| **Config** | `config.py` (plain Python) | — | No YAML/TOML overhead; all thresholds, weights, firm lists in one importable module |
| **Validation** | `validate_submission.py` | provided | Run before every submission; catches format errors before upload |
| **Sandbox** | HuggingFace Spaces (Gradio) | — | Free tier, CPU-only, handles file upload, runs rank.py end-to-end on sample |
| **JD parser (LLM)** | Groq (`llama-3.3-70b-versatile`) via `httpx`, in `scorer/jd_parser.py` | — | One call, once, on a ~600-word JD — not per-candidate. Runs only in `precompute.py` (network-allowed, no time limit), never in `rank.py`. Falls back to hand-written `config.py` defaults if `GROQ_API_KEY` is unset or the call fails — the pipeline never hard-depends on it |

### Why NOT these alternatives

| Skipped Tool | Reason |
|---|---|
| `scikit-learn` TF-IDF / BM25 | This is exactly what the sample submission does — keyword frequency matching rewards keyword stuffers |
| OpenAI / Anthropic / Groq API **during ranking** | Violates compute constraints (no network during ranking). We do call Groq, but only once in `precompute.py` to parse the JD — never per-candidate, never in `rank.py` |
| LLM API call **per candidate** (any provider, any phase) | 100K calls would not fit any time budget and is explicitly the failure mode the spec's compute constraints are designed to filter out (see Section 3: "running an LLM call for each of 100,000 candidates will not fit the 5-minute CPU budget, even if the model runs locally") |
| Local LLM (Llama, Mistral) for per-candidate scoring | 100K × LLM inference on CPU = hours, not minutes |
| `faiss` for ANN search | Unnecessary — we're comparing 100K vectors against 5 query vectors, not doing k-NN search; plain matrix multiply is faster and simpler |
| `spaCy` / `nltk` | Overhead not justified; regex + Python string ops handle keyword extraction adequately |
| `polars` instead of `pandas` | Marginal speed gain at this scale; pandas is more universally understood for Stage 4/5 reviewers |
| `xgboost` LTR | No labeled training pairs available; ground truth is hidden |
| `dask` | 100K records fit in RAM; distributed compute adds complexity with no benefit |

### Dependency install

```
# Step 1 — CPU-only torch first (required for Python 3.13 compatibility):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2 — everything else:
sentence-transformers>=3.0.0   # provides both SentenceTransformer (bi-encoder) and CrossEncoder (reranker)
pandas>=2.2.0
numpy>=1.26.0
pyarrow>=14.0.0
python-dateutil>=2.9.0
langdetect>=1.0.9
```

Total install size (CPU-only torch): ~1.2 GB  
Model download (`all-MiniLM-L6-v2`): 22 MB (cached after first run)

---

## 1. Problem Framing

The challenge is not "find candidates with AI keywords." The sample submission proves this: it ranks an HR Manager and a Content Writer in the top 5 because they have 8–9 AI skills listed. That is keyword matching. It is wrong.

The real problem is:

> Given 100,000 candidate profiles and a job description for a Senior AI Engineer (founding team, Redrob AI, Pune/Noida), identify the 100 candidates who would actually be worth a recruiter's time — ranked by genuine fit, not surface-level signal.

The JD is unusually honest about what it needs and what it doesn't. It explicitly names disqualifiers. It explicitly says its ideal candidate "may not use the words RAG or Pinecone." It explicitly warns that behavioral signals matter as much as skills. The solution is architected to respect every one of those signals.

---

## 2. What the JD Actually Needs (Reading Between the Lines)

### 2.1 The Ideal Candidate Profile

| Dimension | What the JD Says | What It Means |
|---|---|---|
| Experience | "5-9 years" | Soft range. Sweet spot is 6-8 years. Won't reject 4 or 10 if signals are strong. |
| AI Experience | "production experience with embeddings-based retrieval" | Must have shipped a retrieval/ranking/search/rec system to real users. Portfolio projects don't count. |
| Company type | "product companies, not pure services" | Time at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini is a negative signal. Entire career there is a disqualifier. |
| Research | "pure research = we will not move forward" | Academic labs and research-only roles without production deployment are disqualifying. Said explicitly. |
| Coding | "this role writes code" | Anyone who has been "Architecture" or "Tech Lead" without writing production code for 18+ months is flagged. |
| Location | "Pune/Noida, open to Hyd/Mumbai/Delhi NCR" | India only. No visa sponsorship. International candidates must be willing to relocate. |
| Availability | "active on Redrob platform" | Explicitly stated in the JD as a hiring criterion. Inactive candidates are not hireable regardless of skill. |

### 2.2 The Explicit Disqualifiers

The JD lists these. They are not suggestions:

1. Entire career at IT services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
2. AI experience consists only of recent (<12 months) LangChain + OpenAI projects with no pre-LLM ML background
3. No production deployment — pure research or academic background
4. Primary expertise is computer vision, speech, or robotics without NLP/IR exposure
5. 18+ months in architecture/tech-lead roles without writing code
6. Title-chasing — switching companies every 1.5 years for a title bump

### 2.3 The Anti-Keyword-Trap

The JD says: "A Tier 5 candidate may not use the words RAG or Pinecone in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."

This is the core design constraint. The system must read career descriptions semantically, not match keywords in the skills list. A candidate who writes "built a search system that finds similar job postings using user behavior signals" is describing an embedding-based retrieval system — without using that terminology.

---

## 3. Dataset Understanding

### 3.1 Schema Summary

Each of the 100,000 candidates has:

```
candidate_id          CAND_XXXXXXX (7-digit format)
profile               headline, summary, location, country, YOE, current title/company/industry
career_history        1-10 roles: company, title, start/end dates, duration_months, description
education             0-5 entries: institution, degree, field, years, tier (tier_1 to tier_4)
skills                0-N entries: name, proficiency (beginner→expert), endorsements, duration_months
certifications        optional
languages             optional
redrob_signals        23 behavioral signals (see Section 3.3)
```

### 3.2 Key Schema Observations

- `education` can be empty (0 entries). Honeypot checks that depend on graduation year must handle this.
- `skills` can be empty. Scoring must fall back to career description parsing.
- `end_date` is null for current roles — use `is_current` flag, not null check alone.
- `offer_acceptance_rate = -1` is a sentinel meaning "no offer history," not "rejected all offers."
- `github_activity_score = -1` is a sentinel meaning "no GitHub linked," not "zero activity."
- `duration_months` in career roles can disagree with `(end_date - start_date)` — treat `duration_months` as stated, compute actual from dates independently for honeypot checks.

### 3.3 The 23 Behavioral Signals and Their Edge Cases

| Signal | Scoring Note | Edge Case |
|---|---|---|
| `last_active_date` | Exponential decay from today | Future dates → clamp to `DATE_TODAY` (`datetime.date.today()`) |
| `recruiter_response_rate` | Only meaningful if candidate has been visible | If `search_appearance_30d < 5`, treat as unknown (0.5), not 0 |
| `offer_acceptance_rate` | Positive signal if > 0.5 | `-1` = unknown, not zero — treat as neutral |
| `github_activity_score` | Positive engineering signal | `-1` = not linked, not zero activity — mild negative for ML engineers only |
| `open_to_work_flag` | Availability signal | True + stale `last_active_date` → `last_active_date` wins |
| `avg_response_time_hours` | Speed signal | Apply log decay — 720 hrs is not 10× worse than 72 hrs |
| `notice_period_days` | Logistical fit | 0 days = immediate, could be unemployed — positive signal either way |
| `expected_salary_range_inr_lpa` | Salary fit | `min > max` = data error, treat as unknown; `min=max=0` = undisclosed |

### 3.4 What the Sample Submission Shows (and Why It's Wrong)

The provided `sample_submission.csv` ranks:
- Rank 1: HR Manager (9 AI core skills)
- Rank 2: HR Manager (9 AI core skills)
- Rank 4: Content Writer (8 AI core skills)
- Rank 12: Marketing Manager (9 AI core skills)

This is pure keyword counting. It has no concept of whether a candidate's career history matches the role. It is the baseline we are built to beat.

---

## 4. Architecture Overview

The solution is split into two phases:

```
Phase 1: precompute.py    (no time limit — run once)
Phase 2: rank.py          (≤5 min, CPU only, no network — the submission step)
```

This split is essential. Generating sentence embeddings for 100K candidates takes ~10 minutes on CPU. The submission spec allows pre-computation to take as long as needed — only the ranking step must fit in 5 minutes. Pre-computed features are saved to `features/features.parquet` (~200 MB).

```
candidates.jsonl.gz
        │
        ▼
[ precompute.py ]
  ├── Ingest & validate
  ├── Honeypot detection
  ├── Hard gate computation
  ├── Rule-based feature extraction (career, skills, behavioral, logistics)
  ├── Sentence embeddings (MiniLM)
  └── Save → features/features.parquet
                    │
                    ▼
             [ rank.py ]
               ├── Load features
               ├── Apply hard gates
               ├── Compute composite scores (vectorized)
               ├── Sort + tiebreak
               ├── Generate reasoning strings
               └── Write submission CSV
```

---

## 5. Phase 1: precompute.py

### 5.1 Ingest

```python
import gzip, json, datetime

DATE_TODAY = datetime.date.today()  # computed once at process start; never hardcode

def load_candidates(path):
    candidates = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # log and skip malformed lines
    return candidates
```

Post-load normalizations applied to every record:
- Clamp `last_active_date` and `signup_date` to `DATE_TODAY` if in the future
- Compute `days_inactive = (DATE_TODAY - last_active_date).days` and store in record
- Sort `career_history` by `start_date` descending (most recent first) — schema does not guarantee order; `career_history[0]` and `build_candidate_text` both depend on this
- If `expected_salary_range_inr_lpa.min > max`, swap them
- If `signup_date > last_active_date`, mark `activity_reliable = False`
- Merge overlapping career date ranges (for accurate ML month calculation)
- Normalize location strings through canonical city map

### 5.2 Honeypot Detection

The spec says ~80 honeypots exist with "subtly impossible profiles." The examples given:
- "8 years of experience at a company founded 3 years ago"
- "Expert proficiency in 10 skills with 0 years used"

Five detection signals are computed. A candidate needs **≥2 signals** to be flagged `honeypot=True` (hard disqualify). A single signal → `honeypot_soft=True` (0.65× score multiplier, stays in pool).

**Signal 1 — Career timeline contradiction:**
For each career role, compute `actual_months = (end_date or DATE_TODAY) - start_date` in months.
If `|stated_duration_months - actual_months| > 18`, the role's timeline is impossible.
If ≥2 roles in the candidate have this contradiction → signal fires.

```python
def timeline_contradiction(role):
    end = parse_date(role["end_date"]) if role["end_date"] else DATE_TODAY
    start = parse_date(role["start_date"])
    actual_months = (end.year - start.year) * 12 + (end.month - start.month)
    return abs(role["duration_months"] - actual_months) > 18
```

**Signal 2 — Expert skills never used:**
Count skills where `proficiency == "expert"` AND `duration_months == 0`.
Threshold: ≥5 such skills → signal fires.

**Signal 3 — Experience vs graduation (only if education exists):**
`min_grad_year = min(e["end_year"] for e in education if e.get("end_year"))`
If `years_of_experience > (2026 - min_grad_year + 3)` → signal fires.
(The +3 allows for gap years and PhD programs.)
If `education` is empty → signal does NOT fire. Missing data ≠ impossible data.

**Signal 4 — Implausible seniority:**
Title contains VP/Director/CTO/Chief AND `years_of_experience < 4` → signal fires.

**Signal 5 — Perfect-and-abandoned profile:**
`profile_completeness_score == 100` AND `days_since_last_active > 730` → signal fires.
(A complete, current-looking profile that no one has touched in 2 years is suspicious.)

### 5.3 Hard Gate Computation

Gates are computed here as boolean/float flags and applied in `rank.py`.

**Gate A — Location (binary):**

```python
TIER1_INDIA_CITIES = {
    "bengaluru", "bangalore", "noida", "gurugram", "gurgaon",
    "delhi", "new delhi", "ncr", "pune", "hyderabad", "mumbai",
    "chennai", "kolkata"
}
TIER2_INDIA_CITIES = { ... }  # other Indian cities

def classify_location(candidate):
    country = candidate["profile"]["country"].lower()
    city = candidate["profile"]["location"].lower()
    relocate = candidate["redrob_signals"]["willing_to_relocate"]

    if country != "india":
        return "intl_no" if not relocate else "intl_open"
    if any(c in city for c in TIER1_INDIA_CITIES):
        return "tier1_india"
    if city == "remote" or "remote" in city:
        return "remote_india"
    return "tier2_india"

# hard_out if: location_class == "intl_no"
```

**Gate B — Consulting fraction (proportional, not binary):**

```python
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra",
    "mphasis", "ltimindtree", "l&t infotech"
}

def consulting_fraction(career_history):
    total_months = sum(r["duration_months"] for r in career_history)
    if total_months == 0:
        return 0.0

    consulting_months = 0
    for role in career_history:
        co = role["company"].lower()
        if any(f in co for f in CONSULTING_FIRMS):
            months = role["duration_months"]
            # Reduction if description suggests product-embedded work
            desc = role.get("description", "").lower()
            if any(w in desc for w in ["embedded at", "client product", "product team", "worked with"]):
                months *= 0.6
            consulting_months += months

    return consulting_months / total_months

# hard_out if fraction == 1.0
# multiplier if fraction in (0.5, 1.0): linear scale 0.5 → 1.0x, 1.0 → 0.5x
def compute_consulting_multiplier(fraction):
    if fraction >= 1.0:
        return 0.0       # caller also sets hard_out = True
    if fraction > 0.5:
        return 1.0 - (fraction - 0.5)   # 0.51 → ~1.0x, 0.99 → ~0.51x
    return 1.0
```

**Gate C — Domain mismatch:**

Look at roles in the last 36 months only.

```python
CV_SPEECH_KEYWORDS = ["computer vision", "image classification", "object detection",
                       "speech recognition", "tts", "asr", "robotics", "lidar"]
NLP_IR_KEYWORDS = ["nlp", "retrieval", "ranking", "recommendation", "search",
                    "embedding", "language model", "text", "information retrieval"]

def domain_mismatch(career_history):
    recent_roles = [r for r in career_history if is_recent(r, months=36)]
    if not recent_roles:
        recent_roles = career_history[:2]  # fallback: most recent 2

    all_text = " ".join(r.get("description", "") + " " + r["title"]
                        for r in recent_roles).lower()

    has_cv_speech = any(k in all_text for k in CV_SPEECH_KEYWORDS)
    has_nlp_ir = any(k in all_text for k in NLP_IR_KEYWORDS)

    return has_cv_speech and not has_nlp_ir

# hard_out if domain_mismatch == True
```

### 5.4 Rule-Based Feature Extraction

#### Career Features

```python
def extract_career_features(career_history, yoe):
    ML_AI_TITLE_KEYWORDS = [
        "machine learning", "ml engineer", "ai engineer", "data scientist",
        "nlp", "deep learning", "research scientist", "applied scientist",
        "recommendation", "search engineer", "ranking", "retrieval"
    ]

    # Merge overlapping intervals before computing ML months
    intervals = [(parse_date(r["start_date"]),
                  parse_date(r["end_date"]) if r["end_date"] else DATE_TODAY)
                 for r in career_history
                 if any(k in r["title"].lower() for k in ML_AI_TITLE_KEYWORDS)]
    ml_months = sum_non_overlapping_months(intervals)

    # Product company fraction
    total_months = sum(r["duration_months"] for r in career_history)
    consulting_mo = sum(r["duration_months"] for r in career_history
                        if is_consulting(r["company"]))
    product_fraction = (total_months - consulting_mo) / max(total_months, 1)

    # Title hopping — only flag post year 3 of career
    career_start = min(parse_date(r["start_date"]) for r in career_history)
    post_yr3_roles = [r for r in career_history
                      if (parse_date(r["start_date"]) - career_start).days > 3*365]
    if len(post_yr3_roles) > 1:
        avg_tenure = sum(r["duration_months"] for r in post_yr3_roles) / len(post_yr3_roles)
        title_hop = avg_tenure < 18
    else:
        title_hop = False

    # Coding gap — flag if most recent title is architecture/management only
    # career_history is sorted most-recent-first (done in ingest), so [0] is correct
    MGMT_KEYWORDS = ["manager", "director", "vp", "head of", "principal", "architect"]
    IC_KEYWORDS = ["engineer", "scientist", "developer", "analyst", "researcher"]
    recent_title = career_history[0]["title"].lower() if career_history else ""
    is_mgmt_title = any(k in recent_title for k in MGMT_KEYWORDS)
    is_ic_title = any(k in recent_title for k in IC_KEYWORDS)
    coding_gap = is_mgmt_title and not is_ic_title
    coding_gap_multiplier = 0.75 if coding_gap else 1.0

    # Startup exposure
    startup_sizes = {"1-10", "11-50", "51-200"}
    founding_team_exp = any(r["company_size"] in startup_sizes for r in career_history)

    # Recent ML role (last 36 months)
    recent_ml = any(
        any(k in r["title"].lower() for k in ML_AI_TITLE_KEYWORDS)
        for r in career_history
        if is_recent(r, months=36)
    )

    # Description depth (proxy for writing ability and profile authenticity)
    desc_word_counts = [len(r.get("description", "").split()) for r in career_history]
    avg_desc_words = sum(desc_word_counts) / max(len(desc_word_counts), 1)

    return {
        "ml_months": ml_months,
        "product_fraction": product_fraction,
        "title_hop": title_hop,
        "coding_gap": coding_gap,
        "coding_gap_multiplier": coding_gap_multiplier,
        "founding_team_exp": founding_team_exp,
        "recent_ml": recent_ml,
        "avg_desc_words": avg_desc_words,
    }
```

#### Skills Features

Skills are grouped into six taxonomy buckets using a synonym map. The map handles the real-world problem that "FAISS" and "vector database" and "pgvector" all refer to the same capability.

```python
SKILL_TAXONOMY = {
    "retrieval": [
        "elasticsearch", "opensearch", "solr", "bm25", "lucene",
        "hybrid search", "sparse retrieval", "inverted index"
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma",
        "pgvector", "scann", "ann", "vector database", "vector store",
        "approximate nearest neighbor"
    ],
    "embeddings": [
        "sentence-transformers", "sentence transformers", "bge", "e5",
        "openai embeddings", "dense retrieval", "semantic search",
        "bi-encoder", "cross-encoder", "bert", "embedding model",
        "text embedding"
    ],
    "llm_ops": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
        "rag", "retrieval augmented", "langchain", "llm", "gpt", "llama",
        "prompt engineering", "instruction tuning"
    ],
    "eval_ranking": [
        "ndcg", "mrr", "map", "mean average precision", "a/b test",
        "offline eval", "online eval", "learning to rank", "ltr",
        "xgboost ltr", "ranknet", "lambdamart", "evaluation framework"
    ],
    "python_mlops": [
        "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
        "fastapi", "flask", "numpy", "pandas", "docker", "kubernetes",
        "mlflow", "airflow", "spark", "kafka"
    ]
}

SKILL_WEIGHTS = {
    "retrieval":   0.25,
    "vector_db":   0.20,
    "embeddings":  0.20,
    "eval_ranking":0.20,
    "python_mlops":0.10,
    "llm_ops":     0.05,
}

PROFICIENCY_WEIGHT = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

def score_skills(skills, avg_desc_words):
    group_scores = {g: 0.0 for g in SKILL_TAXONOMY}

    for skill in skills:
        name = skill["name"].lower()
        proficiency_w = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.2)
        duration_w = min(skill.get("duration_months", 0) / 24, 1.0)
        endorsement_boost = min(skill.get("endorsements", 0) / 20, 0.2)

        skill_score = proficiency_w * duration_w + endorsement_boost

        for group, keywords in SKILL_TAXONOMY.items():
            if any(k in name for k in keywords):
                group_scores[group] = max(group_scores[group], skill_score)
                break

    weighted_sum = sum(group_scores[g] * SKILL_WEIGHTS[g] for g in group_scores)

    # Keyword stuffer detection:
    # fires ONLY if skill count is high AND career descriptions are sparse
    # (prevents penalizing genuinely skilled candidates)
    stuffer_flag = len(skills) > 20 and avg_desc_words < 40
    stuffer_multiplier = 0.70 if stuffer_flag else 1.0

    return weighted_sum, stuffer_flag, stuffer_multiplier


def skill_assessment_bonus(skill_assessment_scores):
    """
    Platform-verified assessments are more trustworthy than self-reported proficiency.
    Extract scores for skills that map to the core taxonomy groups and return a 0-0.10 bonus.
    """
    if not skill_assessment_scores:
        return 0.0
    CORE_TERMS = [
        "retrieval", "elasticsearch", "vector", "embedding", "nlp", "python",
        "machine learning", "ranking", "recommendation", "search"
    ]
    relevant = [v for k, v in skill_assessment_scores.items()
                if any(t in k.lower() for t in CORE_TERMS)]
    if not relevant:
        return 0.0
    avg = sum(relevant) / len(relevant)
    return min(avg / 100 * 0.10, 0.10)  # max 0.10 bonus; added to jd_fit_score in rank.py


EDUCATION_TIER_BONUS = {"tier_1": 0.05, "tier_2": 0.03, "tier_3": 0.01, "tier_4": 0.0, "unknown": 0.0}

def education_tier_bonus(education):
    """Small bonus for tier-1/tier-2 institutions. Founding-team hire context makes this relevant."""
    if not education:
        return 0.0
    return max(EDUCATION_TIER_BONUS.get(e.get("tier", "unknown"), 0.0) for e in education)
```

#### Technical Keyword Overlay

MiniLM sentence embeddings may not capture highly specific technical terms like HNSW, ScaNN, bi-encoder, or cross-encoder at the level a domain expert would. A keyword overlay adds a bonus for these terms appearing in career descriptions.

```python
TECHNICAL_KEYWORDS = [
    "faiss", "hnsw", "scann", "annoy", "bi-encoder", "cross-encoder",
    "dense passage retrieval", "dpr", "colbert", "splade", "bm25",
    "ndcg", "mrr", "map@", "p@10", "offline eval", "online eval",
    "vector index", "embedding drift", "index refresh", "reranking",
    "retrieval pipeline", "hybrid retrieval", "two-tower", "siamese"
]

def tech_keyword_score(career_history):
    all_desc = " ".join(r.get("description", "").lower() for r in career_history)
    hits = sum(1 for kw in TECHNICAL_KEYWORDS if kw in all_desc)
    return min(hits / 5, 1.0)  # 5 hits = full score
```

#### Behavioral Features

```python
def extract_behavioral_features(signals, days_inactive):
    recency_score = math.exp(-days_inactive / 90)  # decay: 0d=1.0, 90d=0.37, 180d=0.14

    # Response rate only meaningful if candidate has been visible to recruiters
    if signals["search_appearance_30d"] < 5:
        response_score = 0.5  # unknown, not penalized
    else:
        response_score = signals["recruiter_response_rate"]

    # Log decay for response time (720h not 10x worse than 72h)
    rt = signals["avg_response_time_hours"]
    response_time_score = math.exp(-rt / 72)

    # GitHub: -1 = unknown (not zero activity)
    gh = signals["github_activity_score"]
    github_score = 0.35 if gh == -1 else gh / 100

    # Notice: sigmoid-like decay
    notice = signals["notice_period_days"]
    notice_score = max(0.0, 1.0 - (notice / 120))

    # Offer: -1 = unknown
    offer = signals["offer_acceptance_rate"]
    offer_score = 0.5 if offer == -1 else offer

    open_bonus = 0.10 if signals["open_to_work_flag"] else 0.0

    interview_score = signals["interview_completion_rate"]

    # saved_by_recruiters_30d: active recruiter demand signal, more reliable than response_rate
    # log scale so 5 saves ≈ 0.46, 20 saves ≈ 1.0
    import math as _math
    saved = signals["saved_by_recruiters_30d"]
    saved_score = min(_math.log1p(saved) / _math.log1p(20), 1.0)

    return {
        "recency_score": recency_score,
        "response_score": response_score,
        "response_time_score": response_time_score,
        "github_score": github_score,
        "notice_score": notice_score,
        "offer_score": offer_score,
        "open_bonus": open_bonus,
        "interview_score": interview_score,
        "saved_score": saved_score,
    }
```

#### Logistics Features

```python
LOCATION_SCORES = {
    "tier1_india":  1.0,
    "tier2_india":  0.80,
    "remote_india": 0.70,
    "intl_open":    0.50,
    "intl_no":      0.0,   # hard gate (removed in rank.py before scoring)
}

WORKMODE_SCORES = {
    "hybrid":   1.0,
    "flexible": 1.0,
    "remote":   0.75,
    "onsite":   0.60,
}

# JD implies ~40-80 LPA for this role (Series A Senior AI Engineer)
JD_SALARY_MIN = 40
JD_SALARY_MAX = 80

def salary_fit(signals):
    s = signals["expected_salary_range_inr_lpa"]
    if s["min"] == 0 and s["max"] == 0:
        return 0.5  # undisclosed → neutral
    cand_min, cand_max = s["min"], s["max"]
    overlap_min = max(cand_min, JD_SALARY_MIN)
    overlap_max = min(cand_max, JD_SALARY_MAX)
    if overlap_max < overlap_min:
        return 0.2  # no overlap → weak negative
    return min((overlap_max - overlap_min) / (JD_SALARY_MAX - JD_SALARY_MIN), 1.0)
```

### 5.5 LLM-Based JD Parsing

Runs once per `precompute.py` invocation, on a single ~600-word JD — not per
candidate, and never in `rank.py`. This is the dynamic "Deep Job Understanding"
piece the hand-written `JD_QUERIES`/`RERANK_JD_TEXT` couldn't provide on their
own: instead of a human pre-deciding the JD's structure, an LLM (Groq,
`llama-3.3-70b-versatile`) extracts it directly from `config.FULL_JD_TEXT_RAW`
(the real JD, condensed only by trimming pure narrative/cultural framing).

**Why this doesn't violate the compute constraints:** Section 3 bans hosted
LLM calls *during the ranking step* specifically because per-candidate LLM
calls don't scale ("running an LLM call for each of 100,000 candidates will
not fit the 5-minute CPU budget"). This is the opposite case — one call, on
one document, in the untimed precompute phase. `rank.py` never imports
`scorer/jd_parser.py` or makes any network call.

**Extraction schema** (`scorer/jd_parser.py::_EXTRACTION_SCHEMA_PROMPT`):

```json
{
  "job_title": "exact job title + company, e.g. 'Senior AI Engineer, Redrob AI'",
  "role_mandate": "1 sentence, ~20-25 words, on what the role actually does",
  "must_have_requirements": ["exactly 4 items, each ~10-15 words, names specific tools/tech"],
  "ideal_candidate_profile": "1 sentence, ~20-25 words",
  "nice_to_have": ["exactly 4 items, each ~8-12 words"],
  "hard_disqualifiers": ["exactly 4 items, each ~10-15 words"]
}
```

This structure feeds the same shape of output the hand-written config
constants already provided — 5 bi-encoder query strings (`_build_queries`)
and one cross-encoder JD text (`_build_rerank_text`) — so nothing downstream
(the bi-encoder, the cross-encoder, `rank.py`) needs to know or care whether
the JD text came from the LLM or the fallback.

**Three bugs found and fixed while building this, in order:**

1. **Token-budget overflow (round 1).** The first prompt asked for "short"
   items but didn't enforce it numerically. Groq wrote full explanatory
   sentences ("Senior engineers who haven't written production code in the
   last 18 months because they moved into architecture or tech-lead-only
   roles" for a single disqualifier item) — 243 words, **480 tokens**, almost
   exactly the same bug as the hand-written version's first draft (824
   tokens), for the same reason: `max_length=512` is shared between the JD
   text and the candidate's text in the cross-encoder pair.

2. **Over-correction.** Fixed round 1 by capping every item at ≤8 words —
   too aggressively. The result (41 words, 85 tokens) was well within budget
   but the rerank scores **collapsed**: `min=0.5003 max=0.5237` across the
   whole reranked pool — the cross-encoder couldn't meaningfully discriminate
   between candidates anymore. Caught this by checking `rerank_score.describe()`
   after a real run rather than just checking token count and assuming it
   worked — a budget-compliant query is useless if it's too sparse to carry
   any signal.

3. **The actual fix: an explicit anchor matters more than length.** Direct
   A/B test — same 15 candidates, same reranker, only the JD text varied —
   isolated the cause: leading with the literal job title ("Senior AI
   Engineer, Redrob AI.") before any paraphrased content took scores from
   collapsed (`0.500-0.501`) to spread (`0.503-0.537`) on the *exact same*
   paraphrased body text. Added `job_title` as its own schema field,
   extracted from the JD verbatim rather than paraphrased, and prepended it
   in `_build_rerank_text`. Final measured result on the full 100K-candidate,
   500-candidate reranked pool: `min=0.545 max=0.730 mean=0.642` — comparable
   to or better than the hand-written fallback's own measured range.
   (Not fully explained — best hypothesis is `bge-reranker-base` needs a
   concrete, literal anchor for "what is this text" before it can usefully
   score an abstract paraphrase against a resume — but reproduced 3 times
   directly, so treated as a hard requirement on the text's structure.)

The length-budget fix from round 1 is still in place as a safety net
(`_RERANK_TEXT_MAX_WORDS = 170`, ~landing around 300 tokens) — items just
target ~10-15 words now instead of ≤8, with worked good/bad examples in the
prompt itself.

**Reliability:** the parsed result is cached to `features/jd_parsed_cache.json`
so repeated `precompute.py` runs don't re-call the API. If `GROQ_API_KEY`
isn't set, the call fails, or the response doesn't match the expected JSON
schema, `get_jd_queries_and_rerank_text()` logs a warning and returns the
hand-written `config.JD_QUERIES`/`config.RERANK_JD_TEXT` instead — the
pipeline has no hard dependency on this API being available or correct.

### 5.6 Semantic Embeddings

**Model:** `all-MiniLM-L6-v2` — 22MB, CPU-optimized, 512 token limit.

**Why this model:** It is the standard CPU-friendly choice for semantic similarity at scale. At 100K candidates it takes ~44 minutes on CPU (pre-compute phase, no time limit) — slower than the 22MB model size would suggest, since candidate texts (headline + summary + skills + multiple role descriptions) run up to the 512-token limit. The embedding dimension is 384, compact enough to keep the full 100K matrix in memory. We initially tried `bge-large-en-v1.5` (1024-dim) and `bge-base-en-v1.5` (768-dim) as the bi-encoder for higher retrieval quality, but both were too slow on CPU for the full 100K pass; MiniLM is the fast first-stage retriever, and quality is recovered by cross-encoding the top 500 (see §5.7).

**JD decomposed into 5 query vectors:**

Rather than embedding the full JD (which loses nuance), we create targeted query embeddings for each key requirement:

```python
JD_QUERIES = [
    # Q1: Production retrieval and search systems
    "Production experience building embedding-based retrieval systems, vector databases, "
    "hybrid search, deployed to real users at scale. Pinecone, Weaviate, Qdrant, FAISS, "
    "Elasticsearch, OpenSearch, dense retrieval, semantic search.",

    # Q2: Ranking evaluation infrastructure
    "Evaluation frameworks for ranking systems. NDCG, MRR, MAP, offline benchmarks, "
    "online A/B testing, recruiter feedback loops, precision at k, offline-to-online "
    "correlation. Systematic measurement of ranking quality.",

    # Q3: Production ML engineering and Python
    "Strong Python, production ML engineering, deploying models to real users, "
    "serving infrastructure, latency-quality tradeoffs, MLOps, monitoring, "
    "data pipelines, feature engineering at scale.",

    # Q4: Startup and product mindset
    "Founding team, early-stage startup, product-company experience, shipping quickly, "
    "learning from users, iterating fast, scrappy engineering, owning the full stack.",

    # Q5: LLMs, reranking, NLP
    "LLM fine-tuning, LoRA, PEFT, retrieval-augmented generation, RAG, reranking, "
    "cross-encoder, bi-encoder, natural language processing, information retrieval, "
    "recommendation systems, learning to rank.",
]
```

**Candidate text construction:**

```python
def build_candidate_text(candidate):
    parts = []

    # Headline + summary always first
    parts.append(candidate["profile"].get("headline", ""))
    parts.append(candidate["profile"].get("summary", ""))

    # Skills next
    for skill in candidate.get("skills", []):
        parts.append(f"{skill['name']} ({skill['proficiency']})")

    # All roles, most-recent-first (career_history is pre-sorted by ingest).
    # No hard cap on role count — let the tokenizer's truncation handle
    # length naturally, since most-recent-first means truncation always
    # drops the oldest, least relevant roles first.
    for role in candidate["career_history"]:
        desc = role.get("description", "").strip()
        if desc:
            parts.append(f"{role['title']} {role['company']} {desc}")

    text = " ".join(p for p in parts if p)
    if len(text.split()) < MIN_CANDIDATE_TEXT_WORDS:  # 15
        return None  # too sparse for meaningful embedding

    # Measured: median candidate text is ~460 tokens, p90 ~670 — truncation
    # (512-token limit) is the common case, not the exception. Most-recent-first
    # ordering ensures truncation cuts old history, never the current/most-recent role.
    return text
```

**Semantic score computation:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed JD queries once
query_embeddings = model.encode(JD_QUERIES, normalize_embeddings=True)

# Embed candidates in batches of 256
candidate_embeddings = model.encode(
    candidate_texts,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Similarity: shape (N_candidates, 5_queries)
similarities = candidate_embeddings @ query_embeddings.T

# Score: blend max (specialist) and mean (generalist)
semantic_scores = 0.5 * similarities.max(axis=1) + 0.5 * similarities.mean(axis=1)
```

The max-mean blend ensures that a candidate who deeply matches one requirement (e.g., pure retrieval expert) scores nearly as well as a generalist who matches all five moderately.

### 5.7 Cross-Encoder Reranking

The bi-encoder above is a *retriever*: it embeds the JD and each candidate independently, so similarity is just a dot product. That is fast relative to a cross-encoder (no full-attention comparison per pair) — though still ~44 min for 100K candidates measured end-to-end (see §10) — but it loses cross-text interaction — it can't tell that "built RAG pipelines for a recruiting platform" is a much stronger match for this exact JD than "used embeddings in a class project," because the two texts never attend to each other.

A cross-encoder fixes this by feeding `(JD_text, candidate_text)` into the model *together*, so every token can attend to every other token. This is far more precise but much slower — it cannot be batched into a single matrix multiply against 100K candidates, so we apply it only to a shortlist.

**Two-stage retrieval (the standard production IR pattern):**

```
Stage 1 (bi-encoder, all 100K)  →  Stage 2 (cross-encoder, top 500)  →  final blend
   fast, approximate filter           slow, precise reranking
```

**Selecting the top-500 pool:** candidates are pre-filtered to exclude `hard_out` and `honeypot`, then ranked by a cheap proxy score so the cross-encoder spends its budget on plausible candidates, not noise:

```python
def _rough_score(r):
    return (
        r["semantic_score"] * 0.5 +
        min(r["ml_months"] / 48.0, 1.0) * 0.3 +
        r["recency_score"] * 0.2
    )
```

**Cross-encoder scoring:**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

pairs = [(FULL_JD_TEXT, candidate_text) for candidate_text in rerank_pool_texts]
raw_scores = reranker.predict(pairs, batch_size=32)

rerank_scores = 1 / (1 + np.exp(-raw_scores))   # sigmoid → [0, 1]
```

`FULL_JD_TEXT` here is a distilled ~150-token version of the real job description (must-haves, hard disqualifiers, explicitly-do-NOT-want list — not the 5 decomposed queries used for the bi-encoder), not the literal full document text.

**A bug we caught and fixed:** `max_length=512` on a `CrossEncoder` applies to the JD+candidate *pair combined*, not to each side independently. An early draft of `RERANK_JD_TEXT` was a faithful but verbose copy of the real JD — 824 tokens on its own. With that draft, the JD text alone blew past the 512-token budget, leaving the tokenizer's pair-truncation almost nothing of the candidate's actual text to score against — every comparison would have been close to meaningless. We measured candidate text lengths directly (median ~460 tokens, p90 ~670 — truncation is the *common* case, not an edge case) and cut `RERANK_JD_TEXT` down to ~150 words / ~150 tokens, keeping only the highest-signal content (must-haves, hard disqualifiers, the explicit "do NOT want" list) and dropping logistics/culture text that's already scored numerically elsewhere (`scorer/logistics.py`, `scorer/behavioral.py`). We also discovered `build_candidate_text()` was ordering career history oldest-first ("for chronological context"), which meant truncation — already the common case — was silently cutting each candidate's *most recent, most relevant* role first. Fixed by switching to most-recent-first ordering, so truncation now drops old history instead of current experience.

**Blending into jd_fit:** candidates in the top-500 pool get `rerank_score > 0`; everyone else keeps `rerank_score = 0.0` (default) and falls back to the bi-encoder score alone:

```python
blended_semantic = np.where(
    df["rerank_score"] > 0,
    0.65 * df["rerank_score"] + 0.35 * df["semantic_score"],
    df["semantic_score"],
)
```

The 500-candidate cutoff and 0.65/0.35 blend weight are deliberately conservative: the cross-encoder only ever *refines* the ranking within the pool the bi-encoder already surfaced as plausible — it cannot promote a candidate the bi-encoder ranked far outside the top 500, which keeps Stage 2 from being able to introduce wild outliers from a single model's quirks.

### 5.8 Save to Parquet

All extracted features are saved per candidate in a single parquet file (~200MB). This is what `rank.py` loads.

This is the actual column list precompute.py writes (matches the `row` dict
built in `_process_candidate()` plus the two columns filled in after batch
embedding/reranking):

```
features.parquet columns (47 total):
  candidate_id
  honeypot, honeypot_soft, hp_signal_count
  location_class, location_score, consulting_frac, consulting_multiplier,
  domain_mismatch, hard_out, salary_score, workmode_score
  yoe, exp_curve_score, ml_months, product_fraction, title_hop,
  coding_gap, coding_gap_multiplier, founding_team_exp, recent_ml, avg_desc_words
  skill_retrieval, skill_vector_db, skill_embeddings, skill_eval_ranking,
  skill_python_mlops, skill_llm_ops, weighted_skill_score
  stuffer_flag, stuffer_multiplier, skill_assessment_bonus, education_tier_bonus
  tech_keyword_score
  recency_score, response_score, response_time_score, github_score,
  notice_score, saved_score, interview_score, offer_score, open_bonus
  current_title, current_company, recruiter_response_rate
  semantic_score
  rerank_score                                # 0.0 unless in top-500 cross-encoder pool
```

Note that `weighted_skill_score` (the 6 skill groups combined with `SKILL_WEIGHTS`)
is computed *once*, here in precompute.py — `rank.py` reads it directly rather
than recombining the 6 raw `skill_*` group columns itself. The raw group
columns are kept in the parquet only so `scorer/reasoning.py` can name the
specific matched groups (e.g. "strong retrieval & vector-DB skills") instead
of just reporting the aggregate score.

---

## 6. Phase 2: rank.py

### 6.1 Sub-Score Computation (vectorized)

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("features/features.parquet")

# Hard gate: remove disqualified candidates
df = df[~df["hard_out"]]
df = df[~df["honeypot"]]
# (~35K–50K candidates remain)

# --- JD Fit Score ---
# weighted_skill_score (the 6 skill groups combined with SKILL_WEIGHTS) was
# already computed once in precompute.py — rank.py reads it directly rather
# than recombining the 6 raw skill_* columns itself.

# Blend bi-encoder with cross-encoder for the top-500 reranked pool;
# everyone else (rerank_score == 0.0, never cross-encoded) keeps the bi-encoder score alone.
blended_semantic = np.where(
    df["rerank_score"] > 0,
    0.65 * df["rerank_score"] + 0.35 * df["semantic_score"],
    df["semantic_score"],
)

jd_base = (
    0.55 * blended_semantic +
    0.25 * df["weighted_skill_score"] +
    0.20 * df["tech_keyword_score"]
)
jd_fit = (jd_base + df["skill_assessment_bonus"]).clip(0, 1)

# --- Career Score ---
ml_component = np.minimum(df["ml_months"] / 48, 1.0)   # 4 yrs ML = full
product_component = df["product_fraction"]

# Experience curve: peaks at 6-8 yrs, slopes off at both ends
def experience_curve(yoe):
    if yoe < 3:   return 0.50
    if yoe < 5:   return 0.75
    if yoe <= 8:  return 1.00
    if yoe <= 10: return 0.90
    if yoe <= 12: return 0.80
    return 0.70

df["exp_curve_score"] = df["yoe"].apply(experience_curve)

trajectory = np.where(
    ~df["title_hop"] & ~df["coding_gap"], 1.00,
    np.where(df["title_hop"] & df["coding_gap"], 0.50,
    np.where(df["title_hop"], 0.70, 0.75))
)
startup_bonus = df["founding_team_exp"].astype(float) * 0.05
recent_ml_bonus = df["recent_ml"].astype(float) * 0.05  # was computed but unused — now applied
edu_bonus = df["education_tier_bonus"]  # 0.0–0.05 from institution tier

career_score = (
    0.35 * ml_component +
    0.30 * product_component +
    0.25 * trajectory +
    0.10 * df["exp_curve_score"] +
    startup_bonus +
    recent_ml_bonus +
    edu_bonus
).clip(0, 1)

# --- Behavioral Score ---
behavioral_score = (
    0.28 * df["recency_score"] +
    0.18 * df["response_score"] +
    0.14 * df["response_time_score"] +
    0.14 * df["github_score"] +
    0.10 * df["notice_score"] +
    0.08 * df["saved_score"] +       # recruiter demand signal (new)
    0.04 * df["interview_score"] +
    0.04 * df["offer_score"] +
    df["open_bonus"]
).clip(0, 1)

# --- Logistics Score ---
logistics_score = (
    0.60 * df["location_score"] +
    0.25 * df["workmode_score"] +
    0.15 * df["salary_score"]
).clip(0, 1)

# --- Composite (base) ---
base_score = (
    0.42 * jd_fit +
    0.32 * career_score +
    0.21 * behavioral_score +
    0.05 * logistics_score
)

# --- Diversity safeguard (see 6.1.1) ---
company = df["current_company"].fillna("")
title   = df["current_title"].fillna("")
has_group = (company != "") & (title != "")
group_key = company + "||" + title
rank_within_group = df.groupby(group_key)["jd_fit_score"].rank(method="first", ascending=False)
diversity_mult = np.where(has_group & (rank_within_group > 2), 0.80, 1.0)

# --- Apply multipliers ---
composite = (
    base_score
    * df["consulting_multiplier"]
    * df["stuffer_multiplier"]
    * df["coding_gap_multiplier"]
    * np.where(df["honeypot_soft"], 0.65, 1.0)
    * diversity_mult
).clip(0, 1)
```

#### 6.1.1 Diversity Safeguard

The dataset contains large clusters of clearly-synthetic filler candidates —
e.g. 485 candidates with `current_company="Pied Piper"` and
`current_title="Accountant"`, 478 with `"Wayne Enterprises"` /
`"Sales Executive"`, recycled fictional company names (Pied Piper, Wayne
Enterprises, Acme Corp, Globex Inc, Dunder Mifflin) paired with completely
irrelevant non-ML titles. These were already going to rank low on
`jd_fit_score` regardless, but as a defensive backstop against any
(`current_company`, `current_title`) cluster — synthetic or genuine —
crowding out diversity in the top 100, only the top `DIVERSITY_GROUP_FREE_COUNT`
(2) candidates per cluster, ranked by `jd_fit_score`, keep full composite
weight; additional members of the same cluster get a `DIVERSITY_PENALTY_MULT`
(0.80) multiplier. Not a hard exclusion — a candidate ranked 3rd in their
cluster can still place if their score is otherwise strong enough to survive
a 20% haircut.

Based on `jd_fit_score` rather than the final `composite`, to avoid a
circular dependency (composite is computed *from* this multiplier).
Candidates with missing `current_company`/`current_title` are exempt (no
meaningful grouping possible) rather than incorrectly clustered together.
Verified on the real 100K dataset before deploying: of 75,502 eligible
candidates, 744 (company, title) clusters exceeded the free-count threshold,
affecting 73,688 candidates total — almost entirely the synthetic filler
clusters described above, confirmed by checking that the top-5 ranked
candidates were unchanged before/after adding this safeguard (the penalty
only pushes already-low-relevance filler further down, it doesn't disturb
genuinely strong matches).

### 6.2 Sort and Tiebreak

```python
df["composite"] = composite

# Sort: primary = composite descending
# Tiebreak for equal scores: candidate_id ascending (REQUIRED by validator)
df_sorted = df.sort_values(
    by=["composite", "candidate_id"],
    ascending=[False, True]
)

top_100 = df_sorted.head(100).reset_index(drop=True)
top_100["rank"] = range(1, 101)
```

The validator explicitly checks that equal scores are broken by `candidate_id` ascending. This is encoded in the validator at lines 139-144. Any other tiebreak order will fail validation.

### 6.3 Weight Justification

| Component | Weight | Why This Weight |
|---|---|---|
| JD Fit | **0.42** | Primary filter — role alignment is the first gate. Capped below 0.5 because over-indexing on semantic similarity reproduces keyword-matching failure modes when vectors happen to align on irrelevant content. |
| Career Depth | **0.32** | The JD explicitly lists production deployment as a hard requirement and consulting/research as disqualifiers. Career features are the most forgery-resistant signal — you can pad a skill list, but you can't fake 4 years of ML titles at product companies. |
| Behavioral | **0.21** | The JD says "active on Redrob platform" is itself a hiring criterion. A perfect-on-paper candidate who hasn't logged in for 6 months isn't actually hireable. Capped at 0.21 because the signals doc explicitly labels the data "simulated" — over-weighting manufactured signals risks gaming synthetic dataset artifacts. |
| Logistics | **0.05** | Location is a hard gate in the pipeline, not a scorer. What remains (salary range, work mode) is negotiable in a founding team hire. |

**New signals added to behavioral:** `saved_by_recruiters_30d` (weight 0.08) displaces equal fractions from `recency_score` (0.30→0.28), `response_score` (0.20→0.18), `response_time_score` (0.15→0.14), `github_score` (0.15→0.14), `interview_score` (0.05→0.04), `offer_score` (0.05→0.04). Total non-bonus weights still sum to 1.0.

**Stability validation:** Three weight configurations are tested in `rank.py`:
- Config A (primary): 0.42 / 0.32 / 0.21 / 0.05
- Config B (career-heavy): 0.35 / 0.38 / 0.22 / 0.05
- Config C (fit-heavy): 0.50 / 0.25 / 0.20 / 0.05

At least 15 of the top-20 candidates must appear across all three configs. If fewer than 15 overlap, a warning is logged (doesn't block output). The goal is to submit a ranking where the top candidates are robust to reasonable weight variation — that is the strongest signal that the system found genuinely good fits.

### 6.4 Reasoning Generation

Reasoning strings are generated deterministically from the candidate's actual profile fields. Every claim is traceable to a specific field in the record. No LLM is called during this step.

The reasoning is evaluated at Stage 4 against six checks:
1. Specific facts from the profile
2. JD connection
3. Honest concerns where applicable
4. No hallucination
5. Variation between candidates
6. Rank consistency (tone matches rank position)

This is the actual `scorer/reasoning.py` logic (simplified slightly for
readability — see the real file for the exact truncation/length-cap details):

```python
_SKILL_GROUP_LABELS = [
    ("skill_retrieval",    "retrieval"),
    ("skill_vector_db",    "vector-DB"),
    ("skill_embeddings",   "embeddings"),
    ("skill_eval_ranking", "eval/ranking"),
    ("skill_python_mlops", "Python/MLOps"),
    ("skill_llm_ops",      "LLM ops"),
]

def _named_skill_groups(row, top_n=2, min_score=0.35):
    scored = [(label, row.get(col, 0)) for col, label in _SKILL_GROUP_LABELS]
    scored = [(l, s) for l, s in scored if s >= min_score]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in scored[:top_n]]

def generate_reasoning(row, rank):
    title = row.get("current_title", "")
    company = row.get("current_company", "")
    yoe = row.get("yoe", 0)
    loc_label = {"tier1_india": "metro India", "tier2_india": "tier-2 India",
                 "remote_india": "remote India", "intl_open": "intl/open"}.get(
                     row.get("location_class", ""), row.get("location_class", ""))

    # fit_score prefers the cross-encoder rerank_score when this candidate
    # was in the top-500 reranked pool; falls back to the bi-encoder score.
    rerank = row.get("rerank_score", 0)
    fit_score = rerank if rerank > 0 else row.get("semantic_score", 0)
    skill_groups = _named_skill_groups(row)

    # --- Strengths, each tied to a specific JD requirement, not generic praise ---
    strengths = []
    if skill_groups:
        strengths.append(f"strong {' & '.join(skill_groups)} skills")
    if row.get("ml_months", 0) >= 48:
        strengths.append(f"{row['ml_months']}mo hands-on ML/AI experience")
    if fit_score >= 0.55:
        strengths.append(f"strong JD fit ({fit_score:.2f})")
    if row.get("tech_keyword_score", 0) >= 0.6:
        strengths.append("deep IR/ranking technical vocabulary (FAISS/NDCG-class terms)")
    if row.get("founding_team_exp"):
        strengths.append("startup/founding-team experience")
    if row.get("recruiter_response_rate", 0) >= 0.70:
        strengths.append(f"responsive to recruiters ({row['recruiter_response_rate']:.0%})")

    # --- Concerns, mapped directly to the JD's explicit "do NOT want" list ---
    concerns = []
    if row.get("coding_gap"):
        concerns.append("most recent title is management-only, JD wants hands-on code")
    if row.get("title_hop"):
        concerns.append("title-hopping pattern JD explicitly screens against")
    if row.get("consulting_multiplier", 1.0) < 0.8:
        frac = row.get("consulting_frac", 0)
        concerns.append(f"{frac:.0%} consulting-firm career, JD prefers product company background")
    if rank >= 76 and not concerns:
        weakest = "JD fit" if fit_score < 0.45 else "overall signal strength"
        concerns.append(f"included as lower-priority filler — {weakest} is the limiting factor here")

    who = f"{yoe:.1f}yr {title}" + (f" at {company}" if company else "")
    s1 = f"{who} | {loc_label}" + (" — " + "; ".join(strengths[:2]) if strengths else "") + "."
    s2 = ("Concern: " + "; ".join(concerns[:2]) + "."
          if concerns else
          ("Also: " + "; ".join(strengths[2:4]) + "." if len(strengths) > 2 else ""))

    return (s1 + (" " + s2 if s2 else "")).strip()[:250]  # hard cap, well within spec
```

Key constraints enforced:
- Only references `current_title`, `current_company`, `yoe`, `location_class` — all guaranteed present in the parquet
- Names the actual matched skill groups (not arbitrary skill names) — `_named_skill_groups()` only returns groups that scored above `_SKILL_GROUP_MIN_SCORE` (0.35)
- Concerns map directly to the JD's explicit "do NOT want" list (title-hoppers, consulting-only career, management-only recent title) rather than generic gap-flagging
- Low-rank "filler" reasoning names the actual weakest signal instead of a bare "filler" label
- Never references company names not in `career_history` or skills not in `skills[]`

### 6.5 Output Validation

```python
# Pre-write assertions
assert len(top_100) == 100, f"Expected 100 rows, got {len(top_100)}"
assert list(top_100["rank"]) == list(range(1, 101)), "Ranks must be 1-100 contiguous"
assert top_100["candidate_id"].nunique() == 100, "Duplicate candidate IDs"

# All IDs must exist in original pool
all_pool_ids = set(load_all_ids("candidates.jsonl.gz"))
output_ids = set(top_100["candidate_id"])
assert output_ids.issubset(all_pool_ids), "Output contains IDs not in candidate pool"

# Scores non-increasing (with tolerance for float precision)
scores = list(top_100["score"])
for i in range(len(scores) - 1):
    assert scores[i] >= scores[i+1] - 1e-9, f"Score not non-increasing at rank {i+1}"

# Write — exactly 4 columns, UTF-8
top_100[["candidate_id", "rank", "score", "reasoning"]].to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)
```

**Critical note:** The submission validator enforces exactly 4 columns (`candidate_id`, `rank`, `score`, `reasoning`) and rejects any CSV with a different header or column count. Extra columns for debugging (jd_fit score breakdown, flags, etc.) must be written to a separate internal file and never included in the submission CSV.

---

## 7. Trap Handling Summary

| Trap Type | How We Handle It |
|---|---|
| **Keyword stuffer** | `stuffer_flag` fires only when skill count > 20 AND avg description word count < 40. Penalizes padding, not genuine skills. |
| **Plain-language Tier 1** | Semantic embedding + tech keyword overlay. A candidate who writes "built a search system that finds similar job postings" will score high on Q1/Q5 query vectors. |
| **Behavioral ghost** | `recency_score` = exp decay. 180-day inactive candidate gets 0.14× on recency. Behavioral score is 21% of composite. |
| **Consulting-only** | Fraction-based, not binary. Entire career at IT services → hard gate. Majority at consulting → multiplier. Partial career → proportional reduction. |
| **Title-hopper** | Only flagged for post-year-3 roles. Early career switching is normal and not penalized. |
| **Honeypot** | Requires ≥2 internal consistency signals. Single signal = soft penalty. Never fires on missing data. |
| **Domain mismatch** | Only fires if recent roles (last 36 months) have CV/speech/robotics AND no NLP/IR. Career transition into AI is allowed. |
| **LangChain-only** | If top skills are LangChain/OpenAI only and no pre-2022 ML career evidence exists, skill scores in core groups (retrieval, eval, embeddings) will naturally be near zero. |
| **Research-only** | ML months at academic/research companies → low `product_fraction`. Career score penalizes this via the product component (30% of career score). |

---

## 8. Evaluation Metric Alignment

The submission is scored by:
- NDCG@10: 50% weight
- NDCG@50: 30% weight
- MAP: 15% weight
- P@10: 5% weight

**Implication:** NDCG@10 and P@10 together carry 55% of the metric weights, and because NDCG@50 is computed over positions 1–50 it is also heavily influenced by the top-10 slots — meaning rank 1–10 precision touches all three dominant metrics. Getting ranks 1-10 right matters more than ranks 51-100 combined.

This shapes two decisions:
1. The tech keyword overlay is specifically designed to surface deep specialists who may be under-represented in semantic similarity scores. Specialists who know HNSW, ScaNN, cross-encoder architecture by name are exactly who would rank in a recruiter's top 10.
2. `recency_score` uses exponential decay (`e^(-days_inactive / 90)`, so 180 days inactive → 0.135) rather than a hard cutoff — a candidate inactive for 180 days is heavily penalized on that one signal but isn't categorically excluded from the top 10, since `behavioral_score` is only 21% of the composite and an otherwise exceptional candidate could still rank highly. This is a deliberate tradeoff, not a guarantee: a hard "180 days inactive = forced low rank" rule was considered and rejected, since the dataset's activity timestamps are simulated and a single hard cutoff risks being an arbitrary, gameable threshold rather than a real signal of hireability.

---

## 9. Repository Structure

```
redrob/
├── precompute.py                  # Phase 1: feature extraction + embeddings + reranking
├── rank.py                        # Phase 2: scoring + output (≤5 min)
├── ingest.py                      # Candidate normalisation (dates, salary, career sort)
├── scorer/
│   ├── __init__.py
│   ├── utils.py                   # Date parsing, consulting-firm match, interval merge
│   ├── career.py                  # Career features: ML months, product fraction, etc.
│   ├── skills.py                  # Skill taxonomy, synonym map, group scoring
│   ├── behavioral.py              # Behavioral signal scoring with edge-case handling
│   ├── semantic.py                # MiniLM bi-encoder + bge-reranker-base cross-encoder
│   ├── honeypot.py                # Internal consistency checks
│   ├── logistics.py               # Location, salary, work mode
│   └── reasoning.py               # Deterministic reasoning string generation
├── config.py                      # All weights, firm lists, thresholds, city map, taxonomy, JD text
├── features/
│   └── features.parquet           # Pre-computed, committed (generated by precompute.py)
├── candidate_schema.json          # Provided JSON Schema reference
├── sample_candidates.json         # Provided sample data
├── sample_submission.csv          # Provided format reference
├── validate_submission.py         # Provided by challenge (unmodified)
├── submission_metadata.yaml       # Filled in from template
├── requirements.txt
├── README.md                      # Setup + reproduction commands
└── SOLUTION.md                    # This file — full architecture and design rationale
```

**README commands:**
```bash
# Phase 1 (one-time pre-computation, no time limit):
python precompute.py --candidates candidates.jsonl.gz --out features/features.parquet

# Phase 2 (submission step, ≤5 min on CPU):
python rank.py --features features/features.parquet --candidates candidates.jsonl.gz --out team_xxx.csv

# Validate before submitting:
python validate_submission.py team_xxx.csv
```

---

## 10. Compute Budget

| Step | Phase | Measured Time |
|---|---|---|
| Load + parse 100K JSONL + rule-based feature extraction (honeypot, gates, skills, career, behavioral) | precompute | ~94s |
| LLM-based JD parsing (Groq, 1 call) — cached after first run, ~0s on reruns | precompute | ~1-2s (first run only) |
| Bi-encoder embedding (100K candidates, MiniLM, batch 512) | precompute | ~44 min |
| Cross-encoder reranking (top 500, bge-reranker-base, batch 32) | precompute | ~3–5 min |
| Save parquet | precompute | ~1s |
| **Total precompute** | | **~48–50 min** |
| Load parquet | rank | ~2s |
| Apply gates (filter to ~40K) | rank | ~1s |
| Composite scoring (vectorized) | rank | ~5s |
| Sort + tiebreak | rank | ~2s |
| Reasoning generation (100 rows) | rank | ~1s |
| Output validation + write | rank | ~2s |
| **Total rank** | | **~15s** |

The ranking step comfortably fits within the 5-minute budget with ~95% margin. The precompute step runs once and its output is committed to the repository (or generated via documented script).

---

## 11. What We're Not Doing (and Why)

Note: we *do* use a hosted LLM (Groq) once, in `precompute.py`, to parse the
JD into structured requirements (§5.5) — that's compliant since it's one
call on one document, not per-candidate, and never touches `rank.py`. The
items below are specifically about *per-candidate* LLM usage, which would
violate the compute constraints regardless of which phase it ran in.

| Approach | Why We Skipped It |
|---|---|
| LLM API calls per candidate (GPT-4, Claude) | Violates compute constraints. No network during ranking. |
| Local LLM inference (Llama, Mistral) | 100K candidates × LLM inference = hours on CPU, not minutes. |
| Learning-to-rank with training data | No labeled training data available. Ground truth is hidden. |
| Fine-tuned embedding model | Requires labeled pairs. We use a pre-trained general-purpose model. |
| Per-candidate GPT reasoning | Would require LLM call per candidate during ranking = disqualified. |
| TF-IDF / BM25 ranking | This is exactly what the sample submission does. It fails on plain-language candidates and rewards keyword stuffers. |

---

## 12. Before/After Comparison

**Sample submission (keyword matching) — Rank 1-5:**
| Rank | Candidate | Title | Why it ranked |
|---|---|---|---|
| 1 | CAND_0004989 | HR Manager | 9 AI skills listed |
| 2 | CAND_0001195 | HR Manager | 9 AI skills listed |
| 3 | CAND_0003114 | ML Engineer | 4 AI skills listed |
| 4 | CAND_0000339 | Content Writer | 8 AI skills listed |
| 5 | CAND_0001082 | HR Manager | 8 AI skills listed |

**Our system (two-stage semantic retrieval, LLM-parsed JD, career/behavioral/logistics) — actual Rank 1-5, from the final run:**
| Rank | Candidate | Score | Reasoning |
|---|---|---|---|
| 1 | CAND_0018499 | 0.8937 | 7.2yr Senior Machine Learning Engineer at Zomato \| metro India — strong vector-DB & eval/ranking skills; 86mo hands-on ML/AI experience. Also: strong JD fit (0.73); deep IR/ranking technical vocabulary (FAISS/NDCG-class terms). |
| 2 | CAND_0081846 | 0.8865 | 6.7yr Lead AI Engineer at Razorpay \| tier-2 India — strong retrieval & vector-DB skills; 79mo hands-on ML/AI experience. Also: strong JD fit (0.73); deep IR/ranking technical vocabulary. |
| 3 | CAND_0046525 | 0.8838 | 6.1yr Senior Machine Learning Engineer at Genpact AI \| metro India — strong embeddings & LLM ops skills; 73mo hands-on ML/AI experience. Also: strong JD fit (0.72); deep IR/ranking technical vocabulary. |
| 4 | CAND_0077337 | 0.8811 | 7.0yr Staff Machine Learning Engineer at Paytm \| tier-2 India — strong retrieval & vector-DB skills; 82mo hands-on ML/AI experience. **Concern: title-hopping pattern JD explicitly screens against.** |
| 5 | CAND_0086022 | 0.8790 | 5.3yr Senior Applied Scientist at Sarvam AI \| metro India — strong vector-DB & embeddings skills; 63mo hands-on ML/AI experience. Also: strong JD fit (0.72); deep IR/ranking technical vocabulary. |

The contrast between these two lists is the core argument of the system: surface-level skill lists predict nothing about genuine role fit. Career history, semantic understanding of role descriptions, and platform behavior together predict hireability — and the system catches its own caveats: rank 4 (Paytm) is a strong candidate on every other dimension but still gets flagged for the exact title-hopping pattern the JD explicitly screens against, rather than being silently rewarded for an otherwise-strong profile.

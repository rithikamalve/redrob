# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Two-phase candidate ranking system: an offline feature/embedding pipeline
(`precompute.py`) feeding a fast, deterministic ranker (`rank.py`) that
produces the submission CSV.

See [SOLUTION.md](SOLUTION.md) for full architecture, design rationale, and
methodology. This file covers setup and reproduction only.

## Quick start

```bash
# 1. Install dependencies (CPU-only torch first — required for Python 3.13)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Drop the dataset into the repo root (not committed — see .gitignore)
#    candidates.jsonl, or candidates.jsonl.gz — both are accepted.

# 3. Run the pipeline
python precompute.py --candidates ./candidates.jsonl --out ./features/features.parquet
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. Validate before submitting
python validate_submission.py submission.csv
```

## The two phases

| Phase | Script | Constraint | What it does |
|---|---|---|---|
| 1 | `precompute.py` | No time limit, network allowed | Parses `candidates.jsonl`, runs all rule-based scorers (honeypot detection, hard gates, career/skills/behavioral/logistics features), embeds all candidates with a bi-encoder (`all-MiniLM-L6-v2`), cross-encodes the top 500 with a reranker (`BAAI/bge-reranker-base`). Writes `features/features.parquet`. Takes ~48–50 minutes on CPU, mostly bi-encoder embedding. |
| 2 | `rank.py` | **≤5 min, CPU only, no network** (the graded constraint) | Loads `features.parquet`, computes the composite score, sorts, runs a stability check, generates reasoning strings, writes the submission CSV. Runs in **~1–2 seconds** — has zero network or GPU-capable imports (only `numpy`/`pandas`/stdlib; verify with `grep -n "^import\|^from" rank.py`). |

`features/features.parquet` is committed to this repo (~4MB) so Stage 3
reproduction can run `rank.py` directly without waiting ~50 minutes for
`precompute.py` to re-download models and re-embed 100K candidates. If you
want to regenerate it from scratch, run `precompute.py` first — it has no
time limit per the spec, it just isn't part of the timed ranking step.

`rank.py` also accepts `--candidates`, matching the exact reproduce-command
form in submission_spec.md Section 10.3:
`python rank.py --candidates ./candidates.jsonl --out ./submission.csv`.
The flag is logged but otherwise unused — everything `rank.py` needs was
already computed offline by `precompute.py` into `features.parquet`, so the
timed ranking step never touches `candidates.jsonl` itself.

## Repository structure

```
.
├── README.md                  # this file — setup & reproduction
├── SOLUTION.md                 # full architecture, design rationale, methodology
├── submission_metadata.yaml    # team/portal metadata (spec Section 10.3)
├── precompute.py                # Phase 1 — offline feature + embedding pipeline
├── rank.py                      # Phase 2 — composite scoring + submission CSV (≤5 min)
├── ingest.py                    # candidate normalisation (dates, salary, career sort)
├── config.py                    # every weight, threshold, keyword list, JD text — single source of truth
├── validate_submission.py       # provided format validator — run before every submission
├── candidate_schema.json        # provided JSON Schema reference
├── requirements.txt
├── scorer/
│   ├── career.py                 # ML months, title-hop, coding-gap, founding-team-exp
│   ├── honeypot.py                # 5-signal honeypot detection
│   ├── logistics.py               # location/salary/work-mode scoring + hard gates
│   ├── behavioral.py              # recency, response, GitHub, notice, saved, interview/offer
│   ├── skills.py                  # skill taxonomy, stuffer detection, assessment/education bonus
│   ├── semantic.py                # bi-encoder embedding + cross-encoder reranking
│   ├── reasoning.py               # deterministic, grounded reasoning string generation
│   └── utils.py                   # date parsing, consulting-firm match, non-overlap merge
├── features/
│   └── features.parquet         # precomputed output (committed — see above)
└── models/                      # downloaded model weights (gitignored — precompute.py fetches them)
```

## Compute environment this was developed and tested on

See `submission_metadata.yaml` for the exact declared environment.

## AI tools used

See `submission_metadata.yaml` → `ai_tools_used` / `ai_usage_summary` for the
honest declaration, per spec Section 10.4.

---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
---

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

# 2. (Optional) Set GROQ_API_KEY for LLM-based JD parsing in precompute.py.
#    Not required — precompute.py falls back to a hand-written JD decomposition
#    if unset. features/jd_parsed_cache.json (committed) already has a cached
#    result, so this isn't needed at all unless you want to re-parse the JD.
#    export GROQ_API_KEY=...        (bash)
#    $env:GROQ_API_KEY = "..."      (PowerShell)

# 3. Drop the dataset into the repo root (not committed — see .gitignore)
#    candidates.jsonl, or candidates.jsonl.gz — both are accepted.

# 4. Run the pipeline
python precompute.py --candidates ./candidates.jsonl --out ./features/features.parquet
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 5. Validate before submitting
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
│   ├── jd_parser.py               # LLM-based JD parsing (Groq, precompute.py only)
│   ├── reasoning.py               # deterministic, grounded reasoning string generation
│   └── utils.py                   # date parsing, consulting-firm match, non-overlap merge
├── features/
│   ├── features.parquet         # precomputed output (committed — see above)
│   └── jd_parsed_cache.json     # cached LLM JD parsing result (committed, avoids re-calling Groq)
└── models/                      # downloaded model weights (gitignored — precompute.py fetches them)
```

## Sandbox demo (spec Section 10.5)

`app.py` is a Gradio app that runs the **real** `precompute.py` + `rank.py`
end-to-end on a small sample (no reimplementation — it subprocess-calls the
literal scripts in this repo). `demo_sample_candidates.jsonl` (250 real
candidates) is the bundled default input — large enough that ≥100 survive
the hard-gate filter, since `rank.py` requires `TOP_N=100` eligible
candidates to produce output.

Run it locally:
```bash
pip install gradio
python app.py
```

To deploy to HuggingFace Spaces:
1. Create a new Space at huggingface.co/new-space — SDK: **Gradio**, hardware: CPU basic (free tier).
2. Push this repo's contents to the Space's git remote (or link the Space to this GitHub repo via Settings → Repository).
3. Spaces auto-installs from `requirements.txt` and runs `app.py`. First run downloads the bi-encoder (~90MB) and reranker (~280MB) models, then takes ~1-2 minutes per pipeline run on the 250-candidate sample.
4. Put the resulting Space URL in `submission_metadata.yaml` → `sandbox_link`.

## Compute environment this was developed and tested on

See `submission_metadata.yaml` for the exact declared environment.

## AI tools used

See `submission_metadata.yaml` → `ai_tools_used` / `ai_usage_summary` for the
honest declaration, per spec Section 10.4.

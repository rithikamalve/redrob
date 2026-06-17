"""
app.py — HuggingFace Spaces sandbox demo (submission_spec.docx Section 10.5).

Runs the REAL precompute.py + rank.py end-to-end on a small candidate sample
via subprocess — not a reimplementation, the literal same code that produces
the full submission. Default sample is demo_sample_candidates.jsonl (250 real
candidates from candidates.jsonl), large enough that >=TOP_N=100 survive the
hard-gate filter (consulting/domain/honeypot), which rank.py requires.

Local run:  python app.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import gradio as gr
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAMPLE = os.path.join(HERE, "demo_sample_candidates.jsonl")


def run_pipeline(uploaded_file, progress=gr.Progress()):
    candidates_path = uploaded_file.name if uploaded_file is not None else DEFAULT_SAMPLE

    with tempfile.TemporaryDirectory() as tmp:
        features_path = os.path.join(tmp, "features.parquet")
        submission_path = os.path.join(tmp, "submission.csv")

        progress(0.05, desc="Running precompute.py (rule-based features + bi-encoder + reranker)...")
        precompute = subprocess.run(
            [sys.executable, os.path.join(HERE, "precompute.py"),
             "--candidates", candidates_path, "--out", features_path],
            capture_output=True, text=True, cwd=HERE,
        )
        if precompute.returncode != 0:
            return None, f"precompute.py failed:\n{precompute.stderr[-3000:]}"

        progress(0.85, desc="Running rank.py (composite scoring + reasoning)...")
        rank = subprocess.run(
            [sys.executable, os.path.join(HERE, "rank.py"),
             "--candidates", candidates_path,
             "--features", features_path, "--out", submission_path],
            capture_output=True, text=True, cwd=HERE,
        )
        if rank.returncode != 0:
            return None, f"rank.py failed:\n{rank.stderr[-3000:]}\n\n{rank.stdout[-1500:]}"

        progress(1.0, desc="Done")
        df = pd.read_csv(submission_path)

        # gr.File needs a path that outlives the TemporaryDirectory context
        out_path = os.path.join(tempfile.gettempdir(), "submission.csv")
        df.to_csv(out_path, index=False)

        log_tail = (precompute.stdout[-1500:] + "\n" + rank.stdout[-1500:]).strip()
        return out_path, df.head(20), log_tail


with gr.Blocks(title="Redrob Candidate Ranker — Sandbox") as demo:
    gr.Markdown(
        """
        # Redrob Candidate Ranker — Sandbox Demo
        Runs the actual `precompute.py` + `rank.py` pipeline end-to-end on a small
        candidate sample (no reimplementation — this calls the real scripts in the repo).

        Upload a `.jsonl` / `.jsonl.gz` file with **at least ~150-200 candidate records**
        (some are filtered by hard gates, and `rank.py` requires >=100 to remain), or
        leave empty to use the bundled 250-candidate sample.

        Full pipeline on this small sample takes roughly 1-2 minutes (vs. ~50 minutes
        for the full 100K-candidate dataset) — most of that is one-time model download
        on a Space's first run.
        """
    )
    file_input = gr.File(label="candidates.jsonl (optional — uses bundled sample if empty)", file_types=[".jsonl", ".gz"])
    run_btn = gr.Button("Run ranking pipeline", variant="primary")
    output_file = gr.File(label="submission.csv (top 100 ranked candidates)")
    output_table = gr.Dataframe(label="Top 20 preview")
    output_log = gr.Textbox(label="Pipeline log (tail)", lines=10)

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input],
        outputs=[output_file, output_table, output_log],
    )

if __name__ == "__main__":
    demo.launch()

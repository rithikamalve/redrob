"""
scorer/ — Feature extraction and scoring modules.

Each sub-module owns one concern and imports constants from config.py only.
Implemented phase-by-phase:
  Phase 2 → honeypot.py  (+ ingest layer in precompute.py)
  Phase 3 → logistics.py, career.py, skills.py, behavioral.py
  Phase 4 → semantic.py  (+ precompute.py main)
  Phase 5 → reasoning.py (+ rank.py)
"""

# Legacy exploratory study

Everything belonging to the exploratory pilot study lives here:

- `generate_chains_vs.py` / `generate_chains_vs.ipynb` — pilot chain generator
  (VS argmax/weighted, 50-seed corpus).
- `exploratory_seeds/` — the pilot seed sentences.
- `run_shard_*`, `launch_shards.py`, `monitor_shards.py`, `watchdog_shards.py`,
  `health_logger.py`, `run_vs_embeddings.py`, `process_vs_embeddings_worker.py`
  — pilot shard and embeddings machinery, tied to the old
  `~/data_export/echo-chain` layout.
- `legacy_old/` — the previous `legacy/` tree (`analysis_01_vs`).

Not used by the confirmatory pipeline in `scripts/confirmatory/`. The
confirmatory code quotes this pilot's prompt templates verbatim only via
`scripts/confirmatory/vs_task_prompts.py`, which carries its own locked copy.

Absolute paths updated on 2026-11-18 to match the current repo layout.

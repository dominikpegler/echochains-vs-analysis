# echochains-vs-analysis

Verbalized-sampling (VS) pipeline and analysis for the EchoChains project.
Tests whether verbalized sampling preserves chain diversity / slows convergence
under iterated LM paraphrasing, using a drift--diffusion decomposition
(mean drift mu vs between-chain diffusion Sigma).

## Setup

1. Create and activate the conda environment:

    ```
    conda env create -n echochain
    conda activate echochain
    ```

2. Install the shared package (see https://github.com/dominikpegler/echochain-core):

    ```
    git clone git@github.com:dominikpegler/echochain-core.git
    cd echochain-core
    pip install -e .
    ```

3. Initialize the pub-utils submodule and install it editable:

    ```
    git submodule update --init --recursive
    pip install -e ./pub-utils
    ```

4. Create the environment file and add API information:

    ```
    cp env_example .env
    ```

    `DATA_DIR` points at the shared data root (default
    `~/Dropbox/data_export/echo-chains`); published outputs go to
    `DATA_DIR_PUB/pub`.

## Pipeline

- `py/generate_chains_vs.py` — generate VS chains (vs_weighted / vs_argmax)
- `run_vs_embeddings.py` / `process_vs_embeddings_worker.py` — embeddings
- `run_shard_worker.py` / `run_shard_driver.py` / `launch_shards.py` /
  `watchdog_shards.py` / `monitor_shards.py` / `health_logger.py` — parallel
  shard pipeline (remote GPU box)
- `write_stats_long.py` — consolidate shards to long CSVs
- `write_vs_axis_csv.py` — per-hop semantic-axis scores
- `compute_sigma.py` / `compute_sigma_exploratory.py` /
  `compute_sigma_sensitivity.py` — between-chain variance (Sigma)
- `py/analysis_01_descriptives_vs.py` — single merged script covering the
  direct, vs_weighted, and vs_argmax arms plus the mu/Sigma analysis:

    ```
    python py/analysis_01_descriptives_vs.py
    ```

## Related repositories

- `echochain-core` — shared `echochain` package
- `pub-utils` — shared publication figure/table helpers (submodule)
- `echochains-generation` — base chain generation and analysis
- `echochains-sim-analysis` — attractor-simulator project

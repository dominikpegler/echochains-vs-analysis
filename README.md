# echochains-vs-analysis

Verbalized-sampling (VS) pipeline and analysis for the EchoChains project.
Tests whether verbalized sampling preserves chain diversity / slows convergence
under iterated LM paraphrasing, using a two-component decomposition
(mean drift mu vs between-chain variance Sigma).

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

## Package layout

The repo is a collection of runnable entrypoints under `scripts/`; there is no
installable library. Everything imports the shared `echochain` package and
`pub_utils` from the editable installs above. Notebooks are standalone
`.ipynb` files; there is no jupytext pairing, and runnable scripts live next to
them as plain `.py` files.

- `scripts/confirmatory/` — confirmatory study setup + collection:
  - `corpus_sampler.py` / `.ipynb` — preregistered seed draw
  - `axes_validation.ipynb` — semantic-axis validation before the freeze
  - `generate_chains_confirmatory.py` (frozen implementation) and
    `generate_chains_confirmatory.ipynb` (local prototype driver)
  - `serve_model.sh` (vLLM launcher: local GPU and ASC-5 via `--slurm`)
  - `qc_chains_confirmatory.py`, `test_confirmatory.py`
  - `vs_task_prompts.py`, `tasks_registry.py` (locked prompts)
- `scripts/preprocessing/` — chain-log exports feeding the analyses:
  `write_stats_confirmatory.py` (incremental walker computing the pilot's
  cumulative metrics and axis scores for the confirmatory logs; GPU required)
- `scripts/analysis/` — analysis notebooks plus
  post-hoc computation from the shard exports (`write_stats_long.py`,
  `write_axis_csv.py`, `compute_sigma.py`, `compute_sigma_exploratory.py`,
  `compute_sigma_sensitivity.py`).
- `scripts/design/` — study design, sample-size and power analysis:
  `design_size.py`, `power_analysis_design.py`,
  `power_analysis_prereg.py` (plus its `.ipynb`).
- `scripts/reporting/` — publication outputs written to `DATA_DIR/pub/`:
  `make_paper_outputs.py`, `make_slide_figures.py`.
- `scripts/legacy_exploratory/` — everything from the exploratory study:
  the pilot chain generator (`generate_chains_vs.py` + notebook), the pilot
  seeds (`exploratory_seeds/`), the shard/embeddings machinery, and the
  old `legacy/` tree (`analysis_01_vs`).

Study assets under `DATA_DIR` are kept apart by a `confirmatory` path component.
Exploratory (pilot) assets keep their existing paths (`pub/descriptives*`,
`data/echodrift_stats/{,vs_weighted,vs_argmax}`) and are frozen; confirmatory
chain logs live under `data/vs_data/logs/<model>/confirmatory/<task>/<condition>/`
(`<model>` may be an HF id with an org prefix), and confirmatory analysis exports
under `data/echodrift_stats/confirmatory/<model>/<config>/<task>/<condition>/`
and `pub/confirmatory/<analysis>/`.

The exploratory descriptive analysis (tables, figures, `metrics_pub.json`) is
`scripts/analysis/analysis_01_descriptives_exploratory.ipynb` (run all cells), or
the staged re-export in `scripts/reporting/make_paper_outputs.py`:

    python scripts/reporting/make_paper_outputs.py --stage all

The confirmatory descriptives (coverage, final-hop tables, variance and drift
summaries) are in `scripts/analysis/analysis_01_descriptives_confirmatory.ipynb`;
they read the exports written by
`scripts/preprocessing/write_stats_confirmatory.py`.

The preregistered H1--H4 hypothesis tests live in
`scripts/analysis/analysis_02_hypothesis_tests.ipynb` (a stub notebook to be
filled in).

## Related repositories

- `echochain-core` — shared `echochain` package
- `pub-utils` — shared publication figure/table helpers (submodule)
- `echochains-generation` — base chain generation and analysis
- `echochains-sim-analysis` — attractor-simulator project

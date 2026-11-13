"""Compatibility entry point for the VS design-size power check.

The design-size calculation was rebuilt in `py/power_analysis_prereg.py` on
2026-10-31 and reworked on 2026-11-13 (H2c dropped; no drift-equivalence
criterion). That notebook now owns the seed-vs-chain comparison, the
pre-registered decision rules, the Benjamini-Hochberg family (32 directional
tests per model), H2a's conjunctive criteria, H2b per-axis power and the
entailment asymmetry, and labeled Multi-k uniform scenarios. The older
script in this path used superseded chi-square approximations and is
intentionally retired.

Run from the repository root:

    DATA_DIR=/home/user/Dropbox/data_export/echochains \
      python py/power_analysis_prereg.py

or execute/sync the paired notebook `power_analysis_prereg.ipynb`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("power_analysis_prereg.py")),
                   run_name="__main__")

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit a MUSICA job with the project account filled in once.
#
# MUSICA rejects jobs that do not name an account, and the account name is
# site/project specific (the numeric project id is usually not the Slurm
# account string). This helper resolves the account in one place and forwards
# everything else to sbatch:
#
#   ./submit_musica.sh serve_generate_qwen_musica_job.sh
#   ./submit_musica.sh --account=p201444 serve_generate_qwen_musica_job.sh
#
# Resolution order: --account flag, $MUSICA_ACCOUNT, then MUSICA_ACCOUNT in
# the repo's .env. The job scripts themselves carry no hardcoded account, so
# the same file works for every project member.
# ---------------------------------------------------------------------------
set -euo pipefail

script_abs="$(readlink -f "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_abs")/../.." && pwd)"

ACCOUNT="${MUSICA_ACCOUNT:-}"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)   ACCOUNT="$2"; shift 2 ;;
    --account=*) ACCOUNT="${1#--account=}"; shift ;;
    -h|--help)
      echo "Usage: submit_musica.sh [--account ACCT] JOB_SCRIPT [sbatch args...]"
      echo "Account: --account, \$MUSICA_ACCOUNT, or MUSICA_ACCOUNT in .env"
      exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done

if [[ -z "$ACCOUNT" && -f "$repo_root/.env" ]]; then
  ACCOUNT="$(python3 - "$repo_root/.env" <<'PY'
import sys
from pathlib import Path
for raw in Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if line.startswith("MUSICA_ACCOUNT="):
        print(line.split("=", 1)[1].strip().strip('"').strip("'"))
        break
PY
)" || true
fi

if [[ -z "$ACCOUNT" ]]; then
  echo "error: no MUSICA project account; pass --account=<name> or set" \
       "MUSICA_ACCOUNT (environment or .env)" >&2
  exit 1
fi
if [[ ${#args[@]} -eq 0 ]]; then
  echo "usage: submit_musica.sh [--account ACCT] JOB_SCRIPT [sbatch args...]" >&2
  exit 2
fi

echo "submitting ${args[0]} with --account=$ACCOUNT"
exec sbatch --account="$ACCOUNT" "${args[@]}"

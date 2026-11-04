#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --array=0-19%10
#SBATCH --job-name=pot-mass-grid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261020_pot/logs/mass-grid-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
eval "$("$root/.venv/bin/python" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys,shlex
t=json.load(open("/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261020_pot/mass_grid.json"))[int(sys.argv[1])]
for k,v in t.items(): print(f"{k}={shlex.quote(str(v))}")
PY
)"
echo "episode=$episode mass=$pot_mass output=$output"
"$root/.venv/bin/python" "$root/code/evaluate_pot_mass_variant.py" --episode "$episode" --episode-dir "$episode_dir" --scene "$scene" --checkpoint "$checkpoint" --output "$output" --pot-mass "$pot_mass"

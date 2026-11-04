#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=ep25-pick1v2
#SBATCH --array=0-15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/logs/pick1v2-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
spec=$(python3 -c "import json,sys; print(json.load(open('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/pickup_b1v2/all_specs.json'))[int(sys.argv[1])])" "$SLURM_ARRAY_TASK_ID"); name=$(basename $(dirname "$spec"))_$(basename "$spec" .json)
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u physics_pickup_search.py --spec "$spec" --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/pickup_b1v2/search/$name --samples 128 --iterations 16 --workers 8 --seed 44
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python - <<PYEOF
import json; r=json.load(open('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/pickup_b1v2/search/'+'$name'+'/result.json')); m=r['metrics']['official_style_reference_proxy']
print('RESULT $name', m['mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()})
PYEOF

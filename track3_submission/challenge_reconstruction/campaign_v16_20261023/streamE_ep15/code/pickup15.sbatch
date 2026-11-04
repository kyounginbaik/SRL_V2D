#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=E-pick15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/logs/pick15-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15
spec=$(python3 -c "import json,sys; print(json.load(open('$W/pickup_specs/all_specs.json'))[int(sys.argv[1])])" "$SLURM_ARRAY_TASK_ID"); name=$(basename "$spec" .json)
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u physics_pickup_search_v2.py --spec "$spec" --output $W/pickup/$name --samples 96 --iterations 14 --workers 8 --seed 44
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python - <<PYEOF
import json; r=json.load(open('$W/pickup/$name/result.json')); m=r['metrics']['official_style_reference_proxy']
print('RESULT $name', m['mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()}, 'best', {k:round(v,3) if isinstance(v,float) else v for k,v in r['best'].items() if k!='timing'})
PYEOF

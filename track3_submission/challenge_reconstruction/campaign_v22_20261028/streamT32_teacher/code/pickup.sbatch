#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=B3233-pick
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/pick-%A_%a.log
# usage: sbatch --array=0-N pickup.sbatch SPECS_JSON_LIST OUTROOT [ITER=10] [RESUME_ROOT]   (copy of streamP pickup33.sbatch)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh; cd $R/code
spec=$(python3 -c "import json,sys; print(json.load(open('$1'))[int(sys.argv[1])])" $SLURM_ARRAY_TASK_ID); name=$(basename $spec .json)
RES=""; [ -n "${4:-}" ] && RES="--resume $4/$name/progress.json"
[ -f $2/$name/result.json ] || $PY -B -u physics_pickup_search_v2.py --spec $spec --output $2/$name --samples 96 --iterations ${3:-10} --workers 16 --seed 44 $RES
python3 -c "import json; r=json.load(open('$2/$name/result.json')); m=r['metrics']['official_style_reference_proxy']; print('RESULT $name', m['mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()})"

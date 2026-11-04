#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH -J T33-marginsA
#SBATCH -o /data1/hansol/teacher33/logs/%x-%j.log
# usage: margins33auto.sbatch DISTILL_DIR OUT [SEED] [SAMPLES] [ITERS] -- pick the distilled actor with the lowest worst-object ratio
# max(cm/3, deg/30) from DISTILL_DIR/attempts.json (team actor_grid_c output), then team finger-margin CEM (search_margins_k.py) into OUT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/teacher33/tmp PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; cd /data1/hansol/teacher33/tmp
A=$($PY -c "
import json, sys
r = json.load(open(sys.argv[1] + \"/attempts.json\"))
def wr(x): return max(max(v[\"cm\"] / 3, v[\"deg\"] / 30) for v in x[\"object_errors\"].values())
b = min(r, key=wr); print(b[\"checkpoint\"].rsplit(\"/training/\", 1)[0]); print(\"PICK\", b[\"checkpoint\"], b[\"object_errors\"], round(wr(b), 3), file=sys.stderr)" $1)
echo "ACTOR $A"
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/code/search_margins_k.py --episode 33 --actor $A --output $2 --samples ${4:-64} --iterations ${5:-12} --workers 12 --seed ${3:-3351} 2>&1 | grep --line-buffered -v -E "Warn|warn\("

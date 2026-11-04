#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=E9-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE9_tight/logs/svst-%j.log
# usage: sbatch savestage.sbatch EP TABLE.npz SAVE_OUT STAGING_BASE "note"   (save.sbatch body, then stage.sbatch body with the official-eval object errors)
set -euo pipefail
E9=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE9_tight
bash $E9/code/save.sbatch $1 $2 $3 | tee $3.save.log
OBJ=$(grep '^FINAL' $3.save.log | python3 -c "import sys,json; r=json.loads(sys.stdin.read()[6:])['official_eval_scorer']; print(json.dumps({k: dict(cm=round(v[0],4), deg=round(v[1],3)) for k,v in r.items()}))")
bash $E9/code/stage.sbatch $1 $3/actor $4 "$5" "$OBJ"

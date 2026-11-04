#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=spooncup-chain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/rscc-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EP:?}" "${SRC:?}" "${EPD:?}" "${ROOT:?}" "${WINS:?}" "${SEED:?}"
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
S=$SRC; k=0
for w in $WINS; do T0=${w%-*}; T1=${w#*-}; k=$((k+1)); O=$ROOT/c${k}_${T0}_${T1}
  [ -f $O/result.json ] || /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/retention_spooncup.py --source $S --episode-dir $EPD --output $O --start-s $T0 --stop-s $T1 --cup-weight ${W:-1.0} --seed $((SEED+k)) --scale ${SCALE:-0.5} --spoon-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_0000$EP/wooden_spoon.obj --cup-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_0000$EP/blue_cup.obj --samples 96 --iterations 12 --workers 16
  python3 -c "import json;r=json.load(open('$O/result.json'));print('STAGE',$k,'$w','mp_sr',r['metrics']['reference_proxy_mp_sr'],{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()},'cup_err',round(r['best'].get('spoon_in_cup_error_cm',-1),2))"
  S=$O; done

#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=chain33lr
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/logs/chainlr-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${ROOT:?}" "${WINS:?}" "${SEED:?}"
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
S=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/tail/c1_12.4_16.5; k=0
for w in $WINS; do H=${w%%:*}; r=${w#*:}; T0=${r%-*}; T1=${r#*-}; k=$((k+1)); O=$ROOT/c${k}_${H}_${T0}_${T1}
  if [ $H = L ]; then SIDE=left; PROT=wooden_spoon; else SIDE=right; PROT=blue_cup; fi
  [ -f $O/result.json ] || /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/retention_spooncup.py --source $S --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033 --output $O --start-s $T0 --stop-s $T1 --side $SIDE --protect-object $PROT --cup-weight ${W:-1.5} --seed $((SEED+k)) --scale ${SCALE:-0.5} --spoon-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_000033/wooden_spoon.obj --cup-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_000033/blue_cup.obj --samples 96 --iterations 12 --workers 16 ${FSCALE:+--finger-scale $FSCALE}
  python3 -c "import json;r=json.load(open('$O/result.json'));print('STAGE',$k,'$w','mp_sr',r['metrics']['reference_proxy_mp_sr'],{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()},'cup_err',round(r['best'].get('spoon_in_cup_error_cm',-1),2))"
  S=$O; done

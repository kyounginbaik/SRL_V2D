#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=L8-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE   (unchanged stage_compose_component.py of the stream that built the episode world:
#        ep3/5 streamA_ep35/code, ep6 streamF_margins/ep6/code; teacher = the v22b component's teacher; writes only under BASE)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; L8=$C22/streamL8_3_5_6; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
E=$1; A=$2; B=$3
case $E in
  3) S=$C22/streamA_ep35/code/stage_compose_component.py; TE=$C22/streamA_ep35/ep3/ret/z6s4_r2.5_s21;;
  5) S=$C22/streamA_ep35/code/stage_compose_component.py; TE=$C22/streamA_ep35/ep5/port/O_s4_st1.0_a0.5;;
  6) S=$C22/streamF_margins/ep6/code/stage_compose_component.py; TE=$C22/streamF_margins/ep6/port/O_s4_st0.5_a0.5;;
esac
cd /tmp && $PY -B -u $S --episode $E --distilled $A --base $B --teacher $TE \
  --lineage-file $L8/code/l8res.py --lineage-file $L8/code/l8rel.py --lineage-file $L8/code/plib.py

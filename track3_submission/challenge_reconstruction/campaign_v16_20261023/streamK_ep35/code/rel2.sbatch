#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-rel2
#SBATCH --array=0-17
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/rel2-%A_%a.log
# early release (open when the hand meets the table, wrist frozen) on top of the position-ILC teachers
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
TE=("$K/teachers/ilc2/C_none_gp1.0/iter1 C1" "$K/teachers/ilc2/F_w6.5_none_gp0.7/iter1 F" "$K/teachers/ilc2/B_none_gp0.5/iter0 B4")
RL=("18.3 19.0" "18.3 19.4" "18.45 19.0" "18.45 19.4" "18.6 19.0" "18.6 19.4")
i=$SLURM_ARRAY_TASK_ID; set -- ${TE[$((i/6))]}; T=$1; N=$2; set -- ${RL[$((i%6))]}
cd /tmp && $PY -B -u $A/release_variant.py --teacher $T --episode-dir $EPD --out $K/teachers/rel2/${N}_rel$1_d0.3_lift$2 --t-rel $1 --open-dur 0.3 --t-lift $2 --blend-dur 0.6

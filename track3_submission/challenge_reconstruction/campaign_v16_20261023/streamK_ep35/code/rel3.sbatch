#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-rel3
#SBATCH --array=0-14
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/rel3-%A_%a.log
# hold-in-place (fingers unchanged = open-frac 1.0, wrist frozen until beyond the horizon) and gentle openings
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
F=$K/teachers/ilc2/F_w6.5_none_gp0.7/iter1; B4=$K/teachers/ilc2/B_none_gp0.5/iter0; C1=$K/teachers/ilc2/C_none_gp1.0/iter1
# teacher name t_rel open_dur open_frac t_lift fingers
G=("$F F 18.3 0.3 1.0 30 thumb" "$F F 18.45 0.3 1.0 30 thumb" "$F F 18.6 0.3 1.0 30 thumb" "$B4 B4 18.3 0.3 1.0 30 thumb" "$B4 B4 18.45 0.3 1.0 30 thumb" "$B4 B4 18.6 0.3 1.0 30 thumb" "$C1 C1 18.3 0.3 1.0 30 thumb" "$C1 C1 18.45 0.3 1.0 30 thumb" "$C1 C1 18.6 0.3 1.0 30 thumb" \
   "$F F 18.45 0.3 0.1 19.4 thumb" "$F F 18.45 1.0 0.1 19.8 thumb" "$F F 18.45 1.0 0.1 19.8 thumb,index,middle,ring,pinky" "$C1 C1 18.45 0.3 0.1 19.4 thumb" "$C1 C1 18.45 1.0 0.1 19.8 thumb" "$C1 C1 18.45 1.0 0.1 19.8 thumb,index,middle,ring,pinky")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
TAG=$2_rel$3_d$4_f$5_lift$6_$(echo $7 | tr ',' '-' | cut -c1-9)
cd /tmp && $PY -B -u $A/release_variant.py --teacher $1 --episode-dir $EPD --out $K/teachers/rel3/$TAG --t-rel $3 --open-dur $4 --open-frac $5 --t-lift $6 --blend-dur 0.6 --fingers $7

#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-rel4
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/rel4-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
G=("$K/teachers/ilc4/C1_none_gp0.3/iter0 C13 18.45 0.3" "$K/teachers/ilc4/C1_z_gr0.2_gp0.3/iter1 C1Z 18.45 0.3" "$K/teachers/ilc4/C1_none_gp0.3/iter0 C13 18.45 1.0" "$K/teachers/ilc4/C1_z_gr0.2_gp0.3/iter1 C1Z 18.45 1.0")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $A/release_variant.py --teacher $1 --episode-dir $EPD --out $K/teachers/rel4/$2_rel$3_d$4_thumb --t-rel $3 --open-dur $4 --open-frac 0.1 --t-lift 19.4 --blend-dur 0.6 --fingers thumb

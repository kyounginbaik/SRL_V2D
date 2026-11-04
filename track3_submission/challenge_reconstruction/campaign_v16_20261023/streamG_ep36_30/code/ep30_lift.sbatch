#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=G30-lift
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep30_lift-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
RS=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021; EPDIR=$RS/ep30_v8/episodes/episode_000030
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep30; SRC=$G/ret2/ret_3.2_11.4_s0.5
H=(-0.015 -0.03 -0.045); h=${H[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $RS/ep30_v8/code/extra_lift.py $SRC $G/lift/lift$h $h 2.5 3.5 && $PY -u $RS/ep30_v9/code/release_v9.py $G/lift/lift$h $EPDIR $G/lift/lift${h}_rel11.3 11.3 1.2 0.005 0.06 0.06 body 0.01

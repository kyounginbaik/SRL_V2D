#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35C-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C/logs/distill-%j.log
# usage: sbatch distill.sbatch TEACHER_DIR OUT_DIR [SEED]   (plain LearnedPolicy grid, Stream C actor_grid_c.py, pinned to node01)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; EPD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C/episode/episode_000035
cd /tmp && $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 35 --teacher $1 --output $2 --episode-dir $EPD --workers 8 --seed ${3:-44} --grid 512,0.001,0.00005 1024,0.001,0.00005 2048,0.001,0.00005 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0002 1024,0.01,0.0002 2048,0.01,0.0002 1024,0.03,0.0005

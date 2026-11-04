#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=K8-lot
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/logs/%x-%j.log
# distillation lottery: usage sbatch [--nodelist] lot.sbatch TEACHER_DIR OUT_DIR SEED   (streamC actor_grid_c.py, unchanged; evaluated on the C7 target_final_v5 ep35 world)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; EPD=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC_ep3536/episode/episode_000035
echo host=$(hostname) "$@"
cd /tmp && timeout 2900 $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 35 --teacher $1 --output $2 --episode-dir $EPD --workers 16 --seed $3 --grid 512,0.001,0.00005 1024,0.001,0.00005 2048,0.001,0.00005 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0002 1024,0.01,0.0002 2048,0.01,0.0002 512,0.03,0.0005 1024,0.03,0.0005 2048,0.03,0.0005 1024,0.002,0.0001 1536,0.005,0.0001 768,0.005,0.0001 1536,0.02,0.0003

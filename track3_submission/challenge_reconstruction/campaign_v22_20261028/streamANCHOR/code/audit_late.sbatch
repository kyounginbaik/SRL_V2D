#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=ANC-late
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamANCHOR/logs/%x-%A_%a.log
# late-frame audit (mid / 3/4 / end) for tilt stability: frames from the episode length
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
E=$SLURM_ARRAY_TASK_ID
N=$(python3 -I -c "import numpy as np;print(len(np.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_%06d/object_poses.npz'%$E,allow_pickle=False)['world_from_object']))")
cd /tmp && python3 -I /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamANCHOR/code/anchor_audit.py $E $((N/2)),$((N*3/4)),$((N-12)),$((N-4))

#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=G36-wrap
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_wrap-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36
A=("distilled/seat17.5ilc_rel18.5/actor_c512_p0.01_v0.0002 1.0" "distilled/seat17.5ilc_rel18.5/actor_c512_p0.01_v0.0002 0.5" "distilled/seat17.0_rel18.5/actor_c2048_p0.01_v0.0002 1.0" "distilled/seat17.0_rel18.5/actor_c2048_p0.01_v0.0002 0.5" "distilled/seat17.0_rel18.5/actor_c512_p0.01_v0.0002 1.0" "distilled/seat17.5ilc_rel18.5/actor_c2048_p0.001_v5e-05 1.0"); set -- ${A[$SLURM_ARRAY_TASK_ID]}
N=$(echo $1 | sed 's#distilled/##; s#/#_#g')
cd /tmp && $R/.venv/bin/python -B -u $R/refinement_20261021_ep36/code/wrap_margins.py $G/$1/training/policy.npz $G/wrapped_probe/${N}_s$2 $2

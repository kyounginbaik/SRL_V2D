#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=A2-dist5
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA2_crotch/logs/dist5-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA2_crotch; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
C=("0.01 512" "0.03 512" "0.03 1024" "0.1 1024"); set -- ${C[$SLURM_ARRAY_TASK_ID]}
TE=$W/ep5/ilc/gr0.3_gp0.3_s0.35/iter1; EPD=$R/refinement_20261018/calibrated_scene_candidates/episodes/episode_000005; D=$W/ep5/distilled_ilc1/fg$1_c$2
cd /tmp && $R/.venv/bin/python -B -u $W/code/distill_direct.py --episode 5 --teacher $TE --episode-dir $EPD --out $D --fg $1 --centers $2
$R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code/wrap_keys.py 5 $D $W/ep5/wrapped_ilc1/fg$1_c$2 --keys all --donor $R/refinement_20261020_pot/staging_v3/components/episode_000005/portable/original/policy.npz

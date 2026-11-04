#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=spoon-ilc
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/ilc-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("32 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/preserved_retention/episode_000032/seed81 2.8 11.0 rel_g05 --gain-rot 0.5 --gain-pos 0.6"
   "32 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/preserved_retention/episode_000032/seed81 2.8 11.0 rel_g08 --gain-rot 0.8 --gain-pos 0.8"
   "32 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/preserved_retention/episode_000032/seed81 2.8 11.0 abs_g05 --gain-rot 0.5 --gain-pos 0.6 --absolute"
   "33 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/cup33_retention/seed_72 3.0 14.5 rel_g05 --gain-rot 0.5 --gain-pos 0.6"
   "33 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/cup33_retention/seed_72 3.0 14.5 rel_g08 --gain-rot 0.8 --gain-pos 0.8"
   "33 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/cup33_retention/seed_72 3.0 14.5 abs_g05 --gain-rot 0.5 --gain-pos 0.6 --absolute")
set -- ${G[$SLURM_ARRAY_TASK_ID]}; E=$1; S=$2; T0=$3; T1=$4; N=$5; shift 5
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/spoon_ilc.py $S /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep$E/ilc/$N --t0 $T0 --t1 $T1 --iterations 5 "$@"

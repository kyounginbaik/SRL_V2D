#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=spoon-ilc2
#SBATCH --array=0-7
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/ilc2-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
S32=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/preserved_retention/episode_000032/seed81; S33=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/cup33_retention/seed_72
G=("32 $S32 2.8 11.0 v2_rel_g03 --gain-rot 0.3 --gain-pos 0.3"
   "32 $S32 2.8 11.0 v2_rel_g02s25 --gain-rot 0.2 --gain-pos 0.2 --sigma-s 0.25 --max-pos-m 0.03"
   "32 $S32 2.8 11.0 v2_abs_g03 --gain-rot 0.3 --gain-pos 0.3 --absolute"
   "32 $S32 3.5 9.0 v2_rel_g03_w35 --gain-rot 0.3 --gain-pos 0.3"
   "33 $S33 3.0 14.5 v2_rel_g03 --gain-rot 0.3 --gain-pos 0.3"
   "33 $S33 3.0 14.5 v2_rel_g02s25 --gain-rot 0.2 --gain-pos 0.2 --sigma-s 0.25 --max-pos-m 0.03"
   "33 $S33 3.0 14.5 v2_abs_g03 --gain-rot 0.3 --gain-pos 0.3 --absolute"
   "33 $S33 3.5 12.8 v2_rel_g03_w35 --gain-rot 0.3 --gain-pos 0.3")
set -- ${G[$SLURM_ARRAY_TASK_ID]}; E=$1; S=$2; T0=$3; T1=$4; N=$5; shift 5
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/spoon_ilc.py $S /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep$E/ilc/$N --t0 $T0 --t1 $T1 --iterations 6 "$@"

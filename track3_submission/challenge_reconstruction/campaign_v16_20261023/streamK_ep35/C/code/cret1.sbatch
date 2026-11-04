#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35C-ret
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C/logs/cret1-%A_%a.log
# constant-offset retention CEM (rackscan retention.py) towards the compose-retargeted reference, spoon protected
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$K/C/episode/episode_000035
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
T1=$K/C/teachers/rebased/relt_p0.3_it2
G=("1.6 18.6 1.0 601" "1.6 18.6 2.0 602" "1.6 7.0 2.0 603" "7.0 13.8 2.0 604" "13.8 18.6 2.0 605" "3.0 13.8 1.0 606")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd $R/code && timeout 2900 $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention.py --source $T1 --episode-dir $EPD --output $K/C/teachers/cret1/relt03_ret$1_$2_s$3 --start-s $1 --stop-s $2 --protect-object wooden_spoon --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16

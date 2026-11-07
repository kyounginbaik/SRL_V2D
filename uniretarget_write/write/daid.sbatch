#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
# Official Do As I Do stages on the whole write_on_whiteboard clip, with the board added.
R=/lustre/exaflops/retarget_bench; H=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write; W=/scratch2/exaFLOPs/daid_oakink; G=$R/repos/do-as-i-do/retargeting
unset DISPLAY; export MUJOCO_GL=egl WARP_CACHE_PATH=$W/.warp_cache PYTHONPATH=$G
cd $G; $W/.venv_daid/bin/python $H/daid_write.py; echo rc=$?
cd $H; OMP_NUM_THREADS=2 $R/.venv_spider/bin/python score_any.py daid $G/outputs/sharpa/right/write_wb/0

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
# SPIDER's stock OakInk settings on the whole write_on_whiteboard clip, with the board in the scene.
R=/lustre/exaflops/retarget_bench; H=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write; DD=/scratch2/exaFLOPs/spider_wide/ds; T=write_wb; P=$DD/processed/oakinkv2/sharpa/right/$T
unset DISPLAY; export MUJOCO_GL=egl WARP_CACHE_PATH=$H/.warp_cache HYDRA_FULL_ERROR=1 PATH="$R/.venv_spider/bin:$PATH"; mkdir -p $H/.warp_cache; cd $R/repos/spider
python $R/daid_mjwp/check_pairs.py $P/scene.xml || exit 4
python examples/run_mjwp.py +override=oakinkv2 task=$T data_id=1 robot_type=sharpa embodiment_type=right dataset_dir=$DD ++sanity_check_seconds=0.0 improvement_check_steps=100 show_viewer=false ++viewer=mjcpu
echo MJWP_DONE rc=$?; cd $H; OMP_NUM_THREADS=2 python score_any.py spider $P/1/trajectory_mjwp.npz

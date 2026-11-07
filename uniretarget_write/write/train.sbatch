#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000,asus_6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
# Usage: sbatch -J name train.sbatch TASK ARM SEED [STEPS]. One ablation run: floating Sharpa, CHORD FlashSAC recipe, OakInk clip. Only the contact signal differs by arm.
set -euo pipefail
T=$1; ARM=$2; SEED=$3; STEPS=${4:-100007936}; R=/scratch2/exaFLOPs/srl_yam_run; L=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/fc_write; O=$R/ablate/oak; OUT=runs/$T${TAG:-}/$ARM-s$SEED; mkdir -p $O/$OUT; TMP=/tmp/ab_$SLURM_JOB_ID; mkdir -p $TMP $O/.warp $O/.assets; trap 'rm -rf $TMP' EXIT
case $ARM in
 base)  CUR=${CURR:-write_100m}; V=none;  W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=0.0";;
 assist) CUR=write_100m; V=none; W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=0.0 objective.hand_keypoints.weight=${HW:-0.5} objective.hand_joint_pos.weight=${HJ:-0.5} +bilevel.engine=assist +bilevel.interval=${INTERVAL:-5000192} +bilevel.warmup=${WARMUP:-5000192} +bilevel.eta=${ETA:-0.5} +bilevel.sigma=${SIGMA:-0.01} +bilevel.s_max=${SMAX:-0.75} +bilevel.baseline=${BASE:-0}";;
 track) CUR=${CURR:-write_100m}; V=none; W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=0.0 objective.hand_keypoints.weight=${HW:-0.5} objective.hand_joint_pos.weight=${HJ:-0.5}";;
 pib) CUR=write_100m; V=none; W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=0.0 +bilevel.engine=batch +bilevel.slots=${SLOTS:-16} +bilevel.seg=472 +bilevel.clip=450 +bilevel.interval=${INTERVAL:-5000192} +bilevel.board=/oak/write/board.json +bilevel.single_parquet=/oak/clips/${SINGLE:-write_wb}/none/reference.parquet";;
 practice) CUR=write_100m; V=none;  W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=0.0 +bilevel.engine=practice +bilevel.interval=${INTERVAL:-10000000} +bilevel.warmup=${WARMUP:-10000000} +bilevel.candidates=${CANDS:-10} +bilevel.eval_worlds=16 +bilevel.board=/oak/write/board.json";;
 clean) CUR=ablate_50m_chord; V=clean; W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=10.0";;
 noisy) CUR=ablate_50m_chord; V=noisy; W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=10.0";;
 fc)    CUR=ablate_50m; V=noisy; W="objective.object_keypoints.weight=2.0 objective.contact_wrench_support.weight=0.0 objective.force_closure.enabled=true objective.force_closure.weight=5.0";;
esac
apptainer exec --nv --containall --no-home --home /tmp --env PIB_SEG=${PIB_SEG:-0},FRIC_DR=${FRIC_DR:-},PYTHONUNBUFFERED=1,BOARD_JSON=${BJ-/oak/write/board.json},PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$L:/workspace:ro" -B "$O:/oak" -B "$O/.warp:/tmp/.cache/warp" -B "$O/.assets:/workspace/src/flash_chord/assets/.cache" --pwd /workspace $R/flash_chord_yam.sif python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='/oak/clips/$T/$V/reference.parquet'" logging.mode=disabled "output_dir=/oak/$OUT" "hydra.run.dir=/oak/$OUT/hydra" reset.seed=$SEED training.seed=$SEED training.total_environment_steps=$STEPS training.checkpoint.save_interval_environment_steps=25001984 scene.support=false scene.world_count=2048 training.replay.capacity=4194304 training.replay.observation_storage_dtype=float16 "hydra.searchpath=[file:///oak/conf]" curriculum=$CUR $W ${EXTRA:-}
echo TRAIN_DONE

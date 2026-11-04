#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch -J a38 train_a.sbatch N [seed]   Low level only, same recipe, on the Mink reference.
set -euo pipefail
N=$1; SEED=${2:-42}; EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; E=$M/episodes/$EP; TAG=${TAG:-}; OUT=out/$EP/A$TAG-seed$SEED; PAR=${FROM_POOL:+$(paste -sd, $R/search/$EP/pool.txt)}; PAR=${PAR:-$E/yam/flash/mink/reference.parquet}
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$H/episodes/$EP/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
mkdir -p "$H/$OUT"; TMP=$H/.cache/tmp_t_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT; echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) side=$SIDE obj=$IDX"
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B $R/search:/search --pwd /workspace $R/flash_chord_yam.sif python scripts/train_flash_sac.py experiment=yam_sharpa_flash_sac "task.parquet='$PAR'" "embodiment.config.urdf_path._args_=['$E/yam/flash/yam_sharpa_bimanual.urdf']" logging.mode=disabled "output_dir=$M/$OUT" "hydra.run.dir=$M/$OUT/hydra" reset.seed=$SEED training.seed=$SEED curriculum=${CURR:-yam_frame0_50m} reset.always_reset_to_first_frame=true training.total_environment_steps=50001920 +training.optimization.residual_penalty=${RP:-1} training.replay.capacity=6000000 training.checkpoint.save_interval_environment_steps=12500480
echo TRAIN_A_DONE

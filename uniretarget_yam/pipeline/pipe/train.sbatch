#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch -J t37 train.sbatch N [seed]   Bilevel from the Mink reference with this episode's seeded grasps as candidates.
set -euo pipefail
N=$1; SEED=${2:-42}; EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; E=$M/episodes/$EP; RP=${RP:-1}; R_POOL=/search/$EP/race/${GRIP:-}/flash/reference.parquet; OUT=out/$EP/U${TAG:-}-seed$SEED; [ "$RP" = 1 ] || OUT=out/$EP/U-rp$RP-seed$SEED
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$H/episodes/$EP/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
POOL="[$(paste -sd, $R/search/$EP/pool.txt)]"; [ -n "${NO_SEARCH:-}" ] && { POOL="[]"; OUT=out/$EP/U-nosearch-seed$SEED; }; [ -n "${GRIP:-}" ] && { POOL="[$R_POOL]"; OUT=out/$EP/U-$GRIP${TAG:-}-seed$SEED; }
mkdir -p "$H/$OUT"; TMP=$H/.cache/tmp_t_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT; echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) pool=$POOL side=$SIDE obj=$IDX"
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B $R/search:/search --pwd /workspace $R/flash_chord_yam.sif python scripts/train_flash_sac.py experiment=yam_sharpa_flash_sac "task.parquet='$E/yam/flash/mink/reference.parquet'" "embodiment.config.urdf_path._args_=['$E/yam/flash/yam_sharpa_bimanual.urdf']" logging.mode=disabled "output_dir=$M/$OUT" "hydra.run.dir=$M/$OUT/hydra" reset.seed=$SEED training.seed=$SEED curriculum=${CURR:-yam_frame0_50m} reset.always_reset_to_first_frame=true training.total_environment_steps=50001920 training.replay.capacity=6000000 training.checkpoint.save_interval_environment_steps=12500480 +bilevel.engine=yam_reference +bilevel.interval=10000000 +bilevel.warmup=5000000 +bilevel.candidates=12 +bilevel.eval_worlds=16 +bilevel.side=$SIDE +bilevel.object=$IDX "+bilevel.episode=$E" "+bilevel.code_dir=$M/code" +bilevel.onset_floor=1.4 "+bilevel.pool=$POOL" $([ -f $R/search/$EP/partial.txt ] && echo +bilevel.alone_min=0) +training.optimization.residual_penalty=$RP ${VOCARGS:-}
echo TRAIN_DONE

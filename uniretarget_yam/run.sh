#!/usr/bin/env bash
# Usage: ./run.sh U|A|C [hydra overrides...]
#   U = bilevel: FlashSAC on the human reference, upper level refines it
#   A = FlashSAC on the fixed human reference
#   C = FlashSAC on the fixed refined reference
# Environment: GPU (default 0), IMAGE (default flash_chord:yam), EPISODE (default 37), SEED (default 42), MEM (default 60g), SIDE (default right).
set -euo pipefail
ARM=${1:?U, A or C}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-0}; IMAGE=${IMAGE:-flash_chord:yam}; SEED=${SEED:-42}; MEM=${MEM:-60g}; SIDE=${SIDE:-right}
EP=episode_$(printf %06d "${EPISODE:-37}"); M=/uniretarget_yam; E=$M/episodes/$EP
case $ARM in
  U) REF=fixed; EXTRA=(+bilevel.engine=yam_reference +bilevel.interval=20000000 +bilevel.warmup=30000000 +bilevel.candidates=12 +bilevel.eval_worlds=16 "+bilevel.side=$SIDE" "+bilevel.episode=$E" "+bilevel.code_dir=$M/code");;
  A) REF=fixed; EXTRA=();;
  C) REF=refined; EXTRA=();;
  *) echo "unknown arm: $ARM" >&2; exit 2;;
esac
[ -f "$HERE/episodes/$EP/yam/yam_model.mjb" ] || gunzip -k "$HERE/episodes/$EP/yam/yam_model.mjb.gz"
OUT=out/$EP/arm_$ARM-seed$SEED
# Docker cannot create these mount points inside the read-only source mount.
mkdir -p "$HERE/$OUT" "$HERE/.cache/warp" "$HERE/.cache/assets" "$HERE/flash_chord/src/flash_chord/assets/.cache"
exec docker run --rm --name "ur-yam-$EP-$ARM-$SEED" --gpus "device=$GPU" --user "$(id -u):$(id -g)" --memory "$MEM" --memory-swap "$MEM" \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=4 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$HERE:$M" -v "$HERE/flash_chord:/workspace:ro" -v "$HERE/.cache/warp:/tmp/.cache/warp" -v "$HERE/.cache/assets:/workspace/src/flash_chord/assets/.cache" \
  -w /workspace "$IMAGE" python scripts/train_flash_sac.py experiment=yam_sharpa_flash_sac \
  "task.parquet='$E/yam/flash/$REF/reference.parquet'" "embodiment.config.urdf_path._args_=['$E/yam/flash/yam_sharpa_bimanual.urdf']" \
  logging.mode=disabled "output_dir=$M/$OUT" "hydra.run.dir=$M/$OUT/hydra" "reset.seed=$SEED" "${EXTRA[@]}" "$@"

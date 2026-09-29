#!/usr/bin/env bash
# Re-run actual baseline IK on saved loaded MANO targets, entirely on CPU.
# Original inputs remain read-only. This is not a reconstruction rerun.
set -euo pipefail
REPO=/data1/SRL_V2D
EPISODE=${1:-episode_000041}
INPUT=/data1/video_to_data_challenge/runs/retarget/mesh/$EPISODE
OUTPUT=$REPO/.ablation_lab/retarget_retest/$EPISODE
mkdir -p "$OUTPUT"
docker run --rm --cpus=2 --network=none \
  --user "$(id -u):$(id -g)" --group-add 1234 \
  -e HOME=/tmp -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e HUMAN_MOTION_DATA_DIR=/workspace/e2e/human_motion_data \
  -e ROBOTIC_GROUNDING_INTERMEDIATE_DIR=/workspace/e2e/intermediate \
  -v "$REPO:/workspace/video_to_data:ro" \
  -v "$INPUT:/workspace/e2e:ro" \
  -v "$OUTPUT:/workspace/ablation" \
  --workdir /workspace/video_to_data/robotic_grounding \
  --entrypoint /isaac-sim/python.sh robotic-grounding:latest \
  scripts/retarget/ego_recon_to_sharpa.py \
  --input_dir /workspace/e2e/intermediate/ego_recon/loaded \
  --output_dir /workspace/ablation/processed --sequence_id "$EPISODE" --device cpu --save

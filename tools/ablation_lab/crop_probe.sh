#!/usr/bin/env bash
set -euo pipefail
REPO=/data1/SRL_V2D
OUT=$REPO/.ablation_lab/crop_probe
mkdir -p "$OUT"
docker run --rm --cpus=2 --network=none --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 \
  -v "$REPO:/repo:ro" \
  -v "$REPO/reconstruction/data/weights:/weights:ro" \
  -v /data1/video_to_data_challenge/runs/recon/mesh/episode_000041:/input:ro \
  -v "$OUT:/output" --entrypoint python v2d_hamer:latest \
  /repo/tools/ablation_lab/crop_probe.py

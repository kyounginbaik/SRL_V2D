#!/usr/bin/env bash
# GCRL in-hand transition on the episode-36 spoon: train one algorithm arm, then evaluate all start->goal pairs and render the
# feedback video (trained policy in physics + the GOAL hand pose as a transparent light-blue ghost).
# Usage: ./run_gcrl.sh ARM [GPU=0] [STEPS=30081024] [WORLDS=2048]
#   ARM = flashsac  (A: FlashSAC, dense pose reward)
#         ppo       (B: PPO + SimToolReal-style progress-vs-running-best reward + success bonus)
#         tldr      (C: PPO + TLDR-style learned temporal-distance goal reward)
#         eval-<arm> (only evaluate + render the latest checkpoint of that arm)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$HERE/episode_000036; ARM=${1:?arm}; GPU=${2:-0}; STEPS=${3:-30081024}; W=${4:-2048}
IMAGE=${IMAGE:-flash_chord:ur-pinned-22d733}; OLD=/data3/exaFLOPs/uniretarget_tracklower_20261017/gcrl_ep36
MESHES=$(cd "$HERE/../.." && pwd)/robotic_grounding/flash_chord/src/flash_chord/assets/meshes
[ -n "$(ls "$MESHES"/sharpawave 2>/dev/null)" ] || { echo "hand meshes not found in $MESHES (run git lfs pull in the repo)"; exit 1; }
mkdir -p "$ROOT/warp_cache" "$ROOT/asset_cache" "$ROOT/demo" "$ROOT/runs"
DOCK=(docker run --rm --gpus "device=$GPU" --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
      -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYGLET_HEADLESS=true -e XLA_PYTHON_CLIENT_PREALLOCATE=false
      -v "$ROOT:$ROOT" -v "$ROOT/flash_chord:/workspace:ro" -v "$ROOT/warp_cache:/tmp/.cache/warp"
      -v "$MESHES:/workspace/src/flash_chord/assets/meshes:ro" -v "$ROOT/runtime_assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro"
      -v "$ROOT/asset_cache:/workspace/src/flash_chord/assets/.cache" -w /workspace --entrypoint /opt/venv/bin/python "$IMAGE")
# relocate absolute asset paths in the task parquet (once)
"${DOCK[@]}" -c "
import pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
p=Path('$ROOT/frozen/episode_000036/reference.parquet'); t=pq.read_table(p); md=t.schema.metadata
for c in ['object_mesh_paths','object_urdf_paths']:
    v=[[str(x).replace('$OLD', '$ROOT') for x in row] for row in t.column(c).to_pylist()]
    t=t.set_column(t.schema.get_field_index(c), c, pa.array(v, type=t.schema.field(c).type))
pq.write_table(t.replace_schema_metadata(md), p); print('paths ok')"
case $ARM in flashsac|eval-flashsac) S=train_transition.py; O=gcrl_right_seg0; PAT=policy_;; ppo|eval-ppo) S=train_transition_ppo.py; O=gcrl_ppo_right_seg0; PAT=model_;;
             tldr|eval-tldr) S=train_transition_tldr.py; O=gcrl_tldr_right_seg0; PAT=model_;; *) echo "unknown arm $ARM"; exit 1;; esac
if [[ $ARM != eval-* ]]; then "${DOCK[@]}" "$ROOT/code/$S" "$ROOT" right_seg0 "$STEPS" "$W" 2>&1 | tee "$ROOT/runs/${ARM}_train.log" | grep --line-buffered -E "^\[gcrl|training progress|Error|Traceback" || true; fi
A=${ARM#eval-}; P=$(ls "$ROOT/runs/$O/${PAT}"*.safetensors 2>/dev/null | sort -t_ -k2 -n | tail -1); [ -n "$P" ] || { echo "no checkpoint in $ROOT/runs/$O"; exit 1; }
"${DOCK[@]}" "$ROOT/code/eval_transition_video.py" "$ROOT" right_seg0 "$P" "$ROOT/demo/gcrl_${A}_transitions.mp4" 2>&1 | grep -E "^\[eval\]|Error|Traceback" || true
echo "video: $ROOT/demo/gcrl_${A}_transitions.mp4   numbers: $ROOT/demo/gcrl_${A}_transitions.json"

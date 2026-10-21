#!/usr/bin/env bash
# Isolated grasp-stability test (Dexonomy-style), episode 37 spoon: 21 Dexonomy grasps + the UniRetarget-refined grasp with its
# trained lower policy. Per panel: hand placed at the grasp (fingers open, no table, gravity off) -> fingers close -> settle ->
# force ramps 0 -> 3 m*g along +x/-x/+y/-y/+z/-z; the panel turns red with the breaking force when the spoon slips (> 2 cm / 15 deg).
# Usage:  ./run_grasp_test.sh [PAGES="1 2 3 4"] [GPU=0]      Output: episode_000037/demo/iso37_p<k>.mp4 (+ .json breaking forces)
# Needs: NVIDIA GPU with a CUDA 12.x driver (RTX 3090 / 4090 / A100 / L40S all fine; < 8 GB VRAM), Docker + NVIDIA container toolkit,
#        the flash_chord image (build once:  docker build -t flash_chord:ur-pinned-22d733 -f episode_000037/flash_chord/docker/Dockerfile episode_000037/flash_chord)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$HERE/episode_000037; PAGES=${1:-"1 2 3 4"}; GPU=${2:-0}
# hand meshes: the bundle copy if present, else the repo's Git-LFS copy (run `git lfs pull` once in the repo)
MESHES=$ROOT/runtime_assets/meshes; [ -d "$MESHES/sharpawave" ] || MESHES=$(cd "$HERE/../.." && pwd)/robotic_grounding/flash_chord/src/flash_chord/assets/meshes
[ -f "$(ls "$MESHES"/sharpawave/*/*.obj "$MESHES"/sharpawave/*.obj 2>/dev/null | head -1)" ] || { echo "hand meshes not found in $MESHES (git lfs pull?)"; exit 1; }
mkdir -p "$ROOT/warp_cache" "$ROOT/asset_cache" "$ROOT/demo"
IMAGE=${IMAGE:-flash_chord:ur-pinned-22d733}; OLD=/data3/exaFLOPs/uniretarget_tracklower_20261017/repaired_000037_ilcauto3/episode_000037
DOCK=(docker run --rm --gpus "device=$GPU" --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
      -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYGLET_HEADLESS=true -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e GS_ISOLATED=1
      -v "$ROOT:$ROOT" -v "$ROOT/flash_chord:/workspace:ro" -v "$ROOT/warp_cache:/tmp/.cache/warp"
      -v "$MESHES:/workspace/src/flash_chord/assets/meshes:ro" -v "$ROOT/runtime_assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro"
      -v "$ROOT/asset_cache:/workspace/src/flash_chord/assets/.cache" -w /workspace --entrypoint /opt/venv/bin/python "$IMAGE")
# 1. relocate absolute paths (reference parquet + run config) from the original server to this folder
"${DOCK[@]}" -c "
import pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
R=Path('$ROOT'); OLD='$OLD'
p=R/'frozen/episode_000037/reference.parquet'; t=pq.read_table(p); md=t.schema.metadata
cols={}
for c in ['object_mesh_paths','object_urdf_paths']:
    v=[[str(x).replace(OLD, str(R)) for x in row] for row in t.column(c).to_pylist()]
    t=t.set_column(t.schema.get_field_index(c), c, pa.array(v, type=t.schema.field(c).type))
pq.write_table(t.replace_schema_metadata(md), p)
c=R/'code/run_mixed.py'; c.write_text(c.read_text().replace(OLD, str(R)))
print('paths relocated')"
# 2. per-page rows (grasp references + the refined reference with its policy)
python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
R = Path(sys.argv[1]); S = R / 'priors/screen'
tags = sorted(p.stem for p in S.glob('t_*_sq03.npz'))
ref = {"label": "lateral tripod|REFINED + trained policy|(UniRetarget result)", "ref": str(R / 'policy/reference_0010.npz'), "policy": str(R / 'policy/policy_2510848.safetensors')}
for k in range(0, len(tags), 6):
    rows = [{"label": t[2:].replace('_sq03', '').replace('_', ' ') + '|(Dexonomy)|open-loop', "ref": str(S / f'{t}.npz'), "policy": None} for t in tags[k:k + 6]] + [ref]
    (R / 'demo' / f'stability_rows_p{k // 6 + 1}.json').write_text(json.dumps(rows, indent=1))
print(len(tags), 'grasp types')
PY
# 3. render the pages
for k in $PAGES; do
  rm -rf "$ROOT/demo/iso37_p${k}_parts"
  "${DOCK[@]}" "$ROOT/code/grasp_stability_video.py" "$ROOT" "$ROOT/priors/screen/dexfit_types.npz" "$ROOT/camera_motion.npz" \
      "$ROOT/demo/iso37_p$k.mp4" "$ROOT/demo/stability_rows_p$k.json" 2>&1 | grep -E "\[iso\] moved|^ROW|GRASP VIDEO|Error|Traceback" || true
done
echo "done: $ROOT/demo/iso37_p*.mp4"

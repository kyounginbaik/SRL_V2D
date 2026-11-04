"""Export a selected20 reconstruction bundle in the compose_target layout so
compose_target/score_vs_compose.py can score policy rollouts against it.
usage: build_recon_target.py BUNDLE OUT"""
import sys, json, shutil
from pathlib import Path
import numpy as np
bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
for ep in sorted(bundle.glob('episode_*')):
    d = out / ep.name; d.mkdir(parents=True, exist_ok=True)
    poses = np.load(ep / 'object_poses.npz'); meshes = json.loads((ep / 'meshes.json').read_text())
    for slot, name in enumerate(str(n) for n in poses['names']):
        np.savez_compressed(d / f'{name}_trajectory.npz', T_world_obj=poses['world_from_object'][:, slot])
        src = ep / meshes[name]['file']
        shutil.copy2(src, d / f'{name}_mesh_metric.obj')
    print(ep.name, [str(n) for n in poses['names']], len(poses['world_from_object']))

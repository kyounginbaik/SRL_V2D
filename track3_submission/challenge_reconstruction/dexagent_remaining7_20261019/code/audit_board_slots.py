#!/usr/bin/env python3
"""Check narrow assembly slots using real surface clearance, without mutation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import numpy as np
import trimesh
import audit_new_collisions as audit

BASE = Path(__file__).resolve().parents[1]
source = BASE / 'hawor_stereo_recon_v2/episode_000025'
poses = np.load(source / 'object_poses.npz')
names = poses['names'].tolist()
transforms = poses['world_from_object']
models = [audit.load_object(source, name) for name in names]
records = []
for giver, recipient in [(0, 1), (1, 0)]:
    vertices = audit.sample(models[giver]['visual'], 2500)
    for frame in sorted(set([0, len(transforms)//2, len(transforms)-1])):
        relative = np.linalg.inv(transforms[frame, recipient]) @ transforms[frame, giver]
        points = vertices @ relative[:3, :3].T + relative[:3, 3]
        depth, _ = audit.penetration(points, models[recipient]['parts'])
        selected = points[depth > .0002]
        selected_depth = depth[depth > .0002]
        if len(selected):
            _, clearance, _ = trimesh.proximity.closest_point(models[recipient]['visual'], selected)
            visible = np.zeros(len(selected), bool)
            for vector in np.vstack([np.eye(3), -np.eye(3)]):
                indices = np.flatnonzero(~visible)
                if not len(indices):
                    break
                directions = np.broadcast_to(vector, (len(indices), 3))
                hit = models[recipient]['visual'].ray.intersects_any(selected[indices]+directions*1e-6,directions)
                visible[indices[~hit]] = True
            false = visible & (clearance > .0005)
        else:
            clearance = np.empty(0); false = np.empty(0, bool)
        records.append(dict(frame=frame,seconds=frame/20.,giver=names[giver],recipient=names[recipient],
            surface_probes=len(points),max_collision_depth_mm=float(depth.max(initial=0)*1000),
            false_probe_count=int(false.sum()),false_depth_max_mm=float(selected_depth[false].max(initial=0)*1000),
            false_visual_clearance_max_mm=float(clearance[false].max(initial=0)*1000),
            false_probe_points_body_m=selected[false].tolist()))
output=BASE/'review/episode25_narrow_slot_audit.json'
output.write_text(json.dumps(dict(scope='narrow-slot surface clearance at unchanged source pair poses',
    free_clearance_threshold_mm=.5,old_generic_clearance_threshold_mm=5.,records=records,
    limitation='Axis-visible free probes only; finite sampling is not an exact collision-equivalence proof.'),indent=2)+'\n')
print(json.dumps(dict(output=str(output),records=[{k:v for k,v in r.items() if k!='false_probe_points_body_m'} for r in records])),flush=True)

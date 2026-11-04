#!/usr/bin/env python3
"""Measure object/hand contacts at the end of immutable episode30 recordings."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import numpy as np
import mujoco

BASE = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
teacher = BASE/'physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1'
rows = []
for report_path in sorted(teacher.glob('release_*/trial*/result.json')):
    report = json.loads(report_path.read_text())
    model = mujoco.MjModel.from_xml_path(report['execution']['scene'])
    recording = np.load(report['execution']['trajectory'])
    samples = []
    for seconds in (11.8, 12.3, 12.8, 13.1):
        k = min(round(seconds*float(recording['fps'])), len(recording['qpos'])-1)
        d = mujoco.MjData(model)
        d.qpos[:] = recording['qpos'][k]
        d.qvel[:] = recording['qvel'][k]
        d.ctrl[:] = recording['ctrl'][k]
        mujoco.mj_forward(model, d)
        contacts = []
        for i,c in enumerate(d.contact):
            names = [model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
            if not any(n.startswith('robot_') for n in names):
                continue
            if not any(n in ('mini_sweeper','plastic_dish_rack') for n in names):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model,d,i,force)
            contacts.append(dict(bodies=names,force_n=float(force[0]),penetration_mm=max(0.,-float(c.dist)*1000)))
        samples.append(dict(seconds=seconds,contacts=contacts))
    row = dict(report=str(report_path),mp_sr=report['metrics']['reference_proxy_mp_sr'],samples=samples)
    rows.append(row)
    print(json.dumps(row),flush=True)
output = BASE/'review/episode30_release_contacts.json'
output.write_text(json.dumps(rows,indent=2)+'\n')

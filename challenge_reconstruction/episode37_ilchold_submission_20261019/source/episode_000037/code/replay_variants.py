"""Headless zero-action replay of a robot reference in the exact UniRetarget training env (VOC 0).

Composes the training Hydra config from the episode's own run_mixed.py command, applies the
unassisted evaluation composition (no VOC, no resets, full horizon), plays zero residual actions
(targets == reference), and reports MP-SR proxy errors per object for several conditions.

Usage (training image, GPU, cwd=/workspace): python replay_variants.py ROOT VARIANTS.json
"""
import json
import re
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene

ROOT = Path(sys.argv[1])
EP = ROOT.name
cmd = re.search(r"cmd=\[(.*?)\]\n", (ROOT / 'code/run_mixed.py').read_text(), re.S).group(1)
B = ROOT
f = ROOT / 'frozen' / EP
import hashlib  # noqa: E402  (used by the evaluated command list)
import os
a = type('A', (), dict(phase='pilot', seed=44, worlds=1))()
W = int(os.environ.get('REPLAY_WORLDS', '32'))
train = ROOT / 'runs/replay_tmp'
total = interval = search_interval = 1
overrides = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.'))]

VARIANTS = json.loads(Path(sys.argv[2]).read_text())
OVR = f / 'scene_overrides.json'
ORIGINAL_OVR = OVR.read_text()
CONDITIONS = {}
for v in VARIANTS:
    CONDITIONS[v['name']] = dict(ref=v.get('ref'), extra=v.get('extra', []), contact=v.get('contact'))
results = {}
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    for name, cond in CONDITIONS.items():
        ov_json = json.loads(ORIGINAL_OVR)
        if cond.get('contact'):
            ov_json['contact'] = cond['contact']
        OVR.write_text(json.dumps(ov_json, indent=2))
        ov = [o for o in overrides if not (cond['extra'] and o.split('=')[0] in [e.split('=')[0] for e in cond['extra']])] + cond['extra']
        cfg = compose(config_name='train_flash_sac', overrides=ov)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        apply_evaluation_composition(resolved, world_count=W)
        env_cfg = instantiate_typed(OmegaConf.create(resolved['env']), RLEnvConfig)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        kw = {}
        if cond['ref']:
            path = ROOT / cond['ref']
            kw = dict(robot_reference_path=str(path), robot_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with wp.ScopedDevice('cuda:0'):
            setup = setup_scene(parquet=str(f / 'reference.parquet'), control_fps=env_cfg.sim.fps, motion_speed=1.0,
                                embodiment=embodiment, collision=collision, world_count=W,
                                include_support=resolved['scene']['support'], decompose_objects=True,
                                contact_friction=resolved['scene'].get('contact_friction', 1.0), **kw)
            env = RLEnv(setup.scene, setup.reference, config=env_cfg)
            env.reset(frame_id=0)
            env.action_input.zero_()
            F = setup.scene.robot_reference.num_frames
            pos_err, ang_err, wrist_err = [], [], []
            ids = env.command.body_ids_w.numpy().reshape(-1)
            start = env.state_0.body_q.numpy()[ids].copy(); drift = []; palm_err = []
            from flash_chord.embodiments.binding import _frame_poses
            palms = tuple(h.palm_frame for h in setup.scene.layout.hands)
            ref_palm = np.stack([h.wrist_pos_w for h in setup.scene.robot_reference.hands], axis=1)
            for t in range(F - 1):
                env.step()
                actual = env.state_0.body_q.numpy()[ids]
                tp = env.command.body_target_pos_w.numpy().reshape(-1, 3)
                tq = env.command.body_target_quat_w.numpy().reshape(-1, 4)
                pos_err.append(np.linalg.norm(actual[:, :3] - tp, axis=1))
                if t < 25: drift.append(np.linalg.norm(actual[:, :3] - start[:, :3], axis=1))
                pp, _ = _frame_poses(env.state_0.body_q.numpy().reshape(W, -1, 7)[:1], palms)
                palm_err.append(np.linalg.norm(pp[0] - ref_palm[t + 1], axis=1))
                ang_err.append((Rotation.from_quat(actual[:, 3:]) * Rotation.from_quat(tq).inv()).magnitude())
            pos_err, ang_err = np.asarray(pos_err), np.asarray(ang_err)
        objs = list(setup.reference.object_body_names())
        O = len(objs)
        pe = np.asarray(pos_err).reshape(len(pos_err), W, O); ae = np.asarray(ang_err).reshape(len(ang_err), W, O)
        pe = np.nan_to_num(pe, nan=10., posinf=10.); ae = np.nan_to_num(ae, nan=np.pi, posinf=np.pi)
        mpos = pe.mean(0) * 100; mori = np.degrees(ae.mean(0))          # W x O
        world_pass = ((mpos < 3) & (mori < 30)).all(axis=1)
        carry = (pe[60:240, :, objs.index('white_pot_lid')] < .05).all(axis=0) if 'white_pot_lid' in objs else np.zeros(W, bool)
        dr = np.asarray(drift).reshape(len(drift), W, O)
        per = {o: dict(median_position_cm=round(float(np.median(mpos[:, i])), 2), best_position_cm=round(float(mpos[:, i].min()), 2),
                       median_orientation_deg=round(float(np.median(mori[:, i])), 1),
                       pre_contact_drift_cm=round(float(dr[:, :, i].max() * 100), 2)) for i, o in enumerate(objs)}
        palm_err = np.asarray(palm_err)
        per['_hands'] = dict(palm_tracking_err_cm_mean=[round(float(x), 2) for x in palm_err.mean(0) * 100])
        per['_timeline'] = dict(lid_err_cm_per_s_median=[round(float(np.median(pe[k, :, -1]) * 100), 1) for k in range(0, len(pe), 20)],
                                pot_err_cm_per_s_median=[round(float(np.median(pe[k, :, 0]) * 100), 1) for k in range(0, len(pe), 20)])
        results[name] = dict(worlds=W, objects=per, pass_fraction=float(world_pass.mean()), lid_carry_fraction=float(carry.mean()),
                             best_world=dict(index=int(np.argmin(mpos.max(1))), position_cm=mpos[np.argmin(mpos.max(1))].round(2).tolist(),
                                             orientation_deg=mori[np.argmin(mpos.max(1))].round(1).tolist()))
        print(name, json.dumps(results[name]), flush=True)
OVR.write_text(ORIGINAL_OVR)
(ROOT / ('reproduction/' + Path(sys.argv[2]).stem + '_report.json')).write_text(json.dumps(results, indent=2) + '\n')

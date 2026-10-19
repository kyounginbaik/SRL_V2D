"""Headless zero-action replay of a robot reference in the exact UniRetarget training env (VOC 0).

Composes the training Hydra config from the episode's own run_mixed.py command, applies the
unassisted evaluation composition (no VOC, no resets, full horizon), plays zero residual actions
(targets == reference), and reports MP-SR proxy errors per object for several conditions.

Usage (training image, GPU, cwd=/workspace): python replay_reference.py ROOT
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
a = type('A', (), dict(phase='pilot', seed=44, worlds=1))()
train = ROOT / 'runs/replay_tmp'
total = interval = search_interval = 1
overrides = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.'))]

CONDITIONS = {
    'ik_reference': dict(ref=None, extra=[]),
    'dexagent_reference': dict(ref='priors/dexagent_reference.npz', extra=[]),
}
results = {}
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    for name, cond in CONDITIONS.items():
        ov = [o for o in overrides if not (cond['extra'] and o.split('=')[0] in [e.split('=')[0] for e in cond['extra']])] + cond['extra']
        cfg = compose(config_name='train_flash_sac', overrides=ov)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        apply_evaluation_composition(resolved, world_count=1)
        env_cfg = instantiate_typed(OmegaConf.create(resolved['env']), RLEnvConfig)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        kw = {}
        if cond['ref']:
            path = ROOT / cond['ref']
            kw = dict(robot_reference_path=str(path), robot_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with wp.ScopedDevice('cuda:0'):
            setup = setup_scene(parquet=str(f / 'reference.parquet'), control_fps=env_cfg.sim.fps, motion_speed=1.0,
                                embodiment=embodiment, collision=collision, world_count=1,
                                include_support=resolved['scene']['support'], decompose_objects=True,
                                contact_friction=resolved['scene'].get('contact_friction', 1.0), **kw)
            env = RLEnv(setup.scene, setup.reference, config=env_cfg)
            env.reset(frame_id=0)
            env.action_input.zero_()
            F = setup.scene.robot_reference.num_frames
            pos_err, ang_err, wrist_err = [], [], []
            ids = env.command.body_ids_w.numpy().reshape(-1)
            start = env.state_0.body_q.numpy()[ids].copy(); drift = []
            for t in range(F - 1):
                env.step()
                actual = env.state_0.body_q.numpy()[ids]
                tp = env.command.body_target_pos_w.numpy().reshape(-1, 3)
                tq = env.command.body_target_quat_w.numpy().reshape(-1, 4)
                pos_err.append(np.linalg.norm(actual[:, :3] - tp, axis=1))
                if t < 25: drift.append(np.linalg.norm(actual[:, :3] - start[:, :3], axis=1))
                ang_err.append((Rotation.from_quat(actual[:, 3:]) * Rotation.from_quat(tq).inv()).magnitude())
            pos_err, ang_err = np.asarray(pos_err), np.asarray(ang_err)
        objs = list(setup.reference.object_body_names())
        per = {o: dict(mean_position_cm=round(float(pos_err[:, i].mean() * 100), 2),
                       mean_orientation_deg=round(float(np.degrees(ang_err[:, i].mean())), 1),
                       final_position_cm=round(float(pos_err[-1, i] * 100), 2))
               for i, o in enumerate(objs)}
        passed = all(v['mean_position_cm'] < 3 and v['mean_orientation_deg'] < 30 for v in per.values())
        for i, o in enumerate(objs): per[o]['pre_contact_drift_cm'] = round(float(np.max(np.asarray(drift)[:, i]) * 100), 2)
        results[name] = dict(objects=per, mp_sr_proxy_pass=passed,
                             note='raw (no rigid alignment); archive scorer additionally aligns object0 at frame 0')
        print(name, json.dumps(results[name]), flush=True)
(ROOT / 'reproduction/dexagent_replay_report.json').write_text(json.dumps(results, indent=2) + '\n')

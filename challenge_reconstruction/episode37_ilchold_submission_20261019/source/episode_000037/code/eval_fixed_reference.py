"""Does the lower policy drift? Evaluate ONE fixed robot reference with successive lower checkpoints (VOC 0, W worlds).

Usage (training image, GPU, cwd=/workspace): python eval_fixed_reference.py ROOT REFERENCE.npz CKPT1 [CKPT2 ...]
Env: exact training config from ROOT/code/run_mixed.py + unassisted evaluation composition, deterministic actor.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.runtime.jax import to_jax
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

ROOT, REF, CKPTS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
W = int(os.environ.get('EVAL_WORLDS', '64'))
EP = ROOT.name
B, f = ROOT, ROOT / 'frozen' / EP
a = type('A', (), dict(phase='pilot', seed=44, worlds=W))()
train = ROOT / 'runs/eval_tmp'
total = interval = search_interval = W * 4096  # dummy, only satisfies TrainingConfig validation
cmd = re.search(r"cmd=\[(.*?)\]\n", (ROOT / 'code/run_mixed.py').read_text(), re.S).group(1)
overrides = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.', 'task.robot_reference'))]
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    cfg = compose(config_name='train_flash_sac', overrides=overrides)
resolved = OmegaConf.to_container(cfg, resolve=True)
apply_evaluation_composition(resolved, world_count=W)
env_cfg = instantiate_typed(OmegaConf.create(resolved['env']), RLEnvConfig)
training = instantiate_typed(cfg.training, TrainingConfig)
results = {}
with wp.ScopedDevice('cuda:0'):
    setup = setup_scene(parquet=str(f / 'reference.parquet'), control_fps=env_cfg.sim.fps, motion_speed=1.0,
                        embodiment=instantiate_typed(cfg.embodiment, Embodiment), collision=instantiate_typed(cfg.collision, CollisionPolicy),
                        world_count=W, include_support=resolved['scene']['support'], decompose_objects=True,
                        robot_reference_path=str(REF), robot_reference_sha256=hashlib.sha256(REF.read_bytes()).hexdigest())
    objs = list(setup.reference.object_body_names())
    for ck in CKPTS:
        env = RLEnv(setup.scene, setup.reference, config=env_cfg)
        env.reset(frame_id=0); env.capture_step(); env.reset(frame_id=0)
        adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        obs = jnp.array(to_jax(env.reset(frame_id=0), (env.world_count, env.observation_dim)), copy=True)
        state, policy, key, _ = load_policy_for_inference(ck, training, obs, adapter.action_dim, deterministic=True)
        ids = env.command.body_ids_w.numpy().reshape(-1)
        pe, ae = [], []
        A, TP, TQ = [], [], []  # first recorded frame = first stepped frame (command buffers are empty at reset)
        JQ = []
        for t in range(setup.scene.robot_reference.num_frames - 1):
            action, key = policy(state, obs, key)
            tr = adapter.step(action); obs = jnp.array(tr.observation, copy=True)
            act = env.state_0.body_q.numpy()[ids]
            tp = env.command.body_target_pos_w.numpy().reshape(-1, 3); tq = env.command.body_target_quat_w.numpy().reshape(-1, 4)
            A.append(act.copy()); TP.append(tp.copy()); TQ.append(tq.copy())
            JQ.append(np.median(env.state_0.joint_q.numpy().reshape(W, -1)[:, :setup.scene.robot_reference.num_joint_q], axis=0))
            pe.append(np.linalg.norm(act[:, :3] - tp, axis=1))
            ae.append((Rotation.from_quat(act[:, 3:]) * Rotation.from_quat(tq).inv()).magnitude())
        pe = np.nan_to_num(np.asarray(pe).reshape(len(pe), W, -1), nan=1., posinf=1.)
        ae = np.nan_to_num(np.asarray(ae).reshape(len(ae), W, -1), nan=np.pi, posinf=np.pi)
        mpos, mori = pe.mean(0) * 100, np.degrees(ae.mean(0))
        res = dict(frame_success=float(((pe < .1) & (ae < .5)).mean()),
                   mp_sr_pass_fraction=float(((mpos < 3) & (mori < 30)).all(axis=1).mean()),
                   **{f'{o}_median_pos_cm': round(float(np.median(mpos[:, i])), 2) for i, o in enumerate(objs)},
                   **{f'{o}_median_ori_deg': round(float(np.median(mori[:, i])), 1) for i, o in enumerate(objs)})
        # official Track 3 alignment: one rigid transform per world maps the first pose of object slot 0 onto the reference
        A, TP, TQ = (np.asarray(x).reshape(len(x), W, -1, x[0].shape[-1]) for x in (A, TP, TQ))
        apos, aori = np.zeros((W, A.shape[2])), np.zeros((W, A.shape[2])); ERR = {}
        for w in range(W):
            Ra = Rotation.from_quat(TQ[0, w, 0]) * Rotation.from_quat(A[0, w, 0, 3:]).inv()
            off = TP[0, w, 0] - Ra.apply(A[0, w, 0, :3])
            for o in range(A.shape[2]):
                pw = Ra.apply(A[:, w, o, :3]) + off
                qw = Ra * Rotation.from_quat(A[:, w, o, 3:])
                apos[w, o] = np.nan_to_num(np.linalg.norm(pw - TP[:, w, o], axis=1), nan=1.).mean() * 100
                ERR.setdefault(o, []).append(np.nan_to_num(np.linalg.norm(pw - TP[:, w, o], axis=1), nan=1.))
                aori[w, o] = np.degrees(np.nan_to_num((qw * Rotation.from_quat(TQ[:, w, o]).inv()).magnitude(), nan=np.pi).mean())
        res.update(aligned_mp_sr_pass_fraction=float(((apos < 3) & (aori < 30)).all(axis=1).mean()),
                   aligned_sp_sr_pass_fraction=float(((apos < 10) & (np.radians(aori) < .5)).all(axis=1).mean()),
                   aligned_median_pos_cm=round(float(np.median(apos)), 2), aligned_best_pos_cm=round(float(apos.min()), 2),
                   aligned_median_ori_deg=round(float(np.median(aori)), 1),
                   aligned_world_pos_cm=np.round(apos[:, 0], 2).tolist(),
                   aligned_err_cm_per_5frames=np.round(np.median(np.stack(ERR[0]), axis=0)[::5] * 100, 1).tolist())
        # signed object offset per frame with the initial offset removed (translation part of the official alignment)
        sig = (A[:, :, :, :3] - TP) - (A[:1, :, :, :3] - TP[:1])
        np.savez(ROOT / 'reproduction' / f'ilc_offset_{REF.stem}_{Path(ck).stem}.npz', median_offset=np.nanmedian(sig, axis=1),
                 achieved_joint_q_median=np.asarray(JQ), reference_joint_q=setup.scene.robot_reference.joint_q,
                 first_reference_frame=1, worlds=W)
        results[Path(ck).name] = res
        print(Path(ck).name, json.dumps(res), flush=True)
out = ROOT / 'reproduction' / f'fixed_reference_drift_{REF.stem}.json'
out.write_text(json.dumps(dict(reference=str(REF), worlds=W, results=results), indent=2) + '\n')

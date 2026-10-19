"""Four-panel demo video (layout of /data3/exaFLOPs/epi19.mp4), one shared top-down camera for the simulated panels:
  1. Original human video (ego cam a)
  2. Reference (kinematic): final refined reference; original video-IK hands as a transparent light-blue ghost
  3. Baseline (physics): video-IK reference, zero residual, no assist (recorded simulation)
  4. Refined (physics): UniRetarget lower policy (deterministic) on the refined reference, no assist (recorded simulation)
Usage (training image, GPU, cwd=/workspace):
  python render_demo.py ROOT FINAL_REF.npz POLICY.safetensors EGO_VIDEO.mp4 OUT.mp4 [CAMERA_MOTION.npz]
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import jax.numpy as jnp
import newton
import numpy as np
import warp as wp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.lifecycle.reset import build_reference_joint_q
from flash_chord.runtime.jax import to_jax
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.visualization.viewer import ViewerConfig, configure_camera, configure_render_style, create_viewer

ROOT, FINAL, POLICY, EGO, OUT = (Path(x) for x in sys.argv[1:6])
CAMFILE = Path(sys.argv[6]) if len(sys.argv) > 6 else None
EP = ROOT.name
f = ROOT / 'frozen' / EP
TMP = OUT.parent / (OUT.stem + '_parts')
TMP.mkdir(parents=True, exist_ok=True)
PW, PH, HEAD = 640, 480, 62

# exact training/evaluation configuration from the run's launcher
B = ROOT
a = type('A', (), dict(phase='pilot', seed=44, worlds=1))()
train = ROOT / 'runs/demo_tmp'
W = 32
total = interval = search_interval = W * 4096  # dummy, satisfies TrainingConfig validation
cmd = re.search(r"cmd=\[(.*?)\]\n", (ROOT / 'code/run_mixed.py').read_text(), re.S).group(1)
overrides = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.', 'task.robot_reference'))]
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    cfg = compose(config_name='train_flash_sac', overrides=overrides)
resolved = OmegaConf.to_container(cfg, resolve=True)
apply_evaluation_composition(resolved, world_count=W)
env_cfg = instantiate_typed(OmegaConf.create(resolved['env']), RLEnvConfig)
training = instantiate_typed(cfg.training, TrainingConfig)


def make_setup(ref_path, worlds=W):
    kw = {} if ref_path is None else dict(robot_reference_path=str(ref_path), robot_reference_sha256=hashlib.sha256(Path(ref_path).read_bytes()).hexdigest())
    return setup_scene(parquet=str(f / 'reference.parquet'), control_fps=env_cfg.sim.fps, motion_speed=1.0,
                       embodiment=instantiate_typed(cfg.embodiment, Embodiment), collision=instantiate_typed(cfg.collision, CollisionPolicy),
                       world_count=worlds, include_support=resolved['scene']['support'], decompose_objects=True, **kw)


def rollout(setup, policy_path):
    """All W worlds; returns body_q (F, W*nb, 7), object body ids (W*O,), object targets (F, W*O, 7)."""
    env = RLEnv(setup.scene, setup.reference, config=env_cfg)
    env.reset(frame_id=0); env.capture_step(); env.reset(frame_id=0)
    frames = [env.state_0.body_q.numpy().copy()]
    F = setup.scene.robot_reference.num_frames
    if policy_path is None:  # zero residual: targets == reference
        env.action_input.zero_()
        for _ in range(F - 1):
            env.step(); frames.append(env.state_0.body_q.numpy().copy())
    else:
        adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        obs = jnp.array(to_jax(env.reset(frame_id=0), (env.world_count, env.observation_dim)), copy=True)
        frames = [env.state_0.body_q.numpy().copy()]
        state, policy, key, _ = load_policy_for_inference(str(policy_path), training, obs, adapter.action_dim, deterministic=True)
        for _ in range(F - 1):
            action, key = policy(state, obs, key)
            obs = jnp.array(adapter.step(action).observation, copy=True)
            frames.append(env.state_0.body_q.numpy().copy())
    ids = env.command.body_ids_w.numpy().reshape(-1)
    return np.stack(frames), ids


def aligned_error(frames, ids, setup):
    """official-style alignment (first pose of object slot 0) per world; mean position error (cm) per world"""
    from scipy.spatial.transform import Rotation
    pos = np.asarray(setup.reference.object_body_pos_w(), np.float64)          # (F, O, 3)
    quat = np.asarray(setup.reference.object_body_quat_w(), np.float64)[..., [1, 2, 3, 0]]
    O = pos.shape[1]
    act = frames[:, ids].reshape(len(frames), -1, O, 7)                          # (F, W, O, 7)
    errs = []
    for w in range(act.shape[1]):
        Ra = Rotation.from_quat(quat[1, 0]) * Rotation.from_quat(act[1, w, 0, 3:]).inv()
        off = pos[1, 0] - Ra.apply(act[1, w, 0, :3])
        e = np.mean([np.linalg.norm(Ra.apply(act[1:, w, o, :3]) + off - pos[1:len(act), o], axis=1).mean() for o in range(O)])
        errs.append(np.nan_to_num(e, nan=9.) * 100)
    return np.asarray(errs)


import os
PANEL = os.environ.get('DEMO_PANEL')
if PANEL is None and not (TMP / 'rollouts.npz').exists():
  with wp.ScopedDevice('cuda:0'):
      final_setup = make_setup(FINAL)
      ik_setup = make_setup(None)
      refined, ids = rollout(final_setup, POLICY)
      baseline, ids_b = rollout(ik_setup, None)
      err_r, err_b = aligned_error(refined, ids, final_setup), aligned_error(baseline, ids_b, ik_setup)
      wr_, wb_ = int(np.argmin(err_r)), int(np.argsort(err_b)[len(err_b) // 2])
      print('refined aligned cm per world', np.round(err_r, 2).tolist(), 'chosen', wr_, flush=True)
      print('baseline aligned cm per world', np.round(err_b, 2).tolist(), 'chosen (median)', wb_, flush=True)
      nb = refined.shape[1] // W
      assert refined.shape[1] == nb * W and all(i // nb == k // (len(ids) // W) for k, i in enumerate(ids)), 'per-world body layout'
      refined, baseline = refined[:, wr_ * nb:(wr_ + 1) * nb], baseline[:, wb_ * nb:(wb_ + 1) * nb]
      np.savez(TMP / 'rollouts.npz', refined=refined, baseline=baseline, err_refined=err_r, err_baseline=err_b, world_refined=wr_, world_baseline=wb_)

_r = np.load(TMP / 'rollouts.npz')
refined, baseline, err_r, err_b = _r['refined'], _r['baseline'], _r['err_refined'], _r['err_baseline']
wr_, wb_ = int(_r['world_refined']), int(_r['world_baseline'])
with wp.ScopedDevice('cuda:0'):
    final_setup = make_setup(FINAL, 1); ik_setup = make_setup(None, 1)
    s, ref = final_setup.scene, final_setup.reference
    F, fps = s.robot_reference.num_frames, s.robot_reference.fps

    # shared top-down camera over the table; image "up" = ego camera's horizontal forward direction
    usda = (f / 'reference_support.usda').read_text()
    table = np.array([float(v) for v in re.search(r'translate = \(([^)]*)\)', usda).group(1).split(',')])
    # original ego camera: per-frame pose (world_from_rgb, OpenCV axes) and vertical FOV from the intrinsics
    from types import SimpleNamespace
    camd = np.load(CAMFILE)
    Twc = camd['world_from_rgb'].astype(np.float64)
    Kc = camd['K']
    VFOV = float(np.degrees(2 * np.arctan(Kc[1, 2] / Kc[1, 1])))
    CP = Twc[:, :3, 3]
    CT = CP + .6 * Twc[:, :3, 2]
    cam, target = CP[0], CT[0]

    def render(name, set_state, robot_only=False):
        flags = s.model.shape_flags.numpy()
        saved = flags.copy()
        if robot_only:
            flags[:s.collision_layout.robot.start] = 0
            flags[s.collision_layout.robot.stop:] = 0
            s.model.shape_flags.assign(flags)
        vc = ViewerConfig(backend='mp4', output_path=str(TMP / f'{name}.mp4'), num_frames=F, video_fps=fps, width=PW, height=PH,
                          headless=True, realtime=False, camera_position=tuple(cam), camera_target=tuple(target),
                          background_color=(0., 0., 0.), support_color=(.96, .62, .33))
        v = create_viewer(vc); v.set_model(s.model); configure_camera(v, vc); configure_render_style(v, vc, scene=s); v.camera.fov = VFOV
        state = s.model.state()
        try:
            for i in range(F):
                configure_camera(v, SimpleNamespace(camera_position=tuple(CP[min(i, len(CP) - 1)]), camera_target=tuple(CT[min(i, len(CT) - 1)]), backend='mp4'))
                set_state(state, i); v.begin_frame(i / fps); v.log_state(state); v.end_frame()
        finally:
            v.close()
            s.model.shape_flags.assign(saved)

    def kinematic(q_robot):
        base = build_reference_joint_q(s, ref)
        q = base.copy(); q[:, :s.robot_reference.num_joint_q] = q_robot

        def set_state(state, i):
            state.joint_q.assign(q[i].astype(np.float32)); state.joint_qd.zero_()
            newton.eval_fk(s.model, state.joint_q, state.joint_qd, state)
        return set_state

    def recorded(body_q):
        def set_state(state, i):
            state.body_q.assign(body_q[i].astype(np.float32))
        return set_state


    if PANEL == 'kinematic_final':
        render(PANEL, kinematic(s.robot_reference.joint_q))
    elif PANEL == 'kinematic_original_hands':
        render(PANEL, kinematic(ik_setup.scene.robot_reference.joint_q), robot_only=True)
    elif PANEL == 'baseline':
        render(PANEL, recorded(baseline))
    elif PANEL == 'refined':
        render(PANEL, recorded(refined))
if PANEL is not None:
    sys.exit(0)
# the GL mp4 viewer records once per process: one subprocess per panel
for _name in ('kinematic_final', 'kinematic_original_hands', 'baseline', 'refined'):
    subprocess.run([sys.executable, __file__, *sys.argv[1:]], check=True, env={**os.environ, 'DEMO_PANEL': _name})
F, fps = int(refined.shape[0]), 20.0


def frames_of(path):
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    cap.release()
    return out


ego = [cv2.resize(im, (PW, PH), interpolation=cv2.INTER_AREA) for im in frames_of(EGO)]
kin, ghost = frames_of(TMP / 'kinematic_final.mp4'), frames_of(TMP / 'kinematic_original_hands.mp4')
base, refi = frames_of(TMP / 'baseline.mp4'), frames_of(TMP / 'refined.mp4')
n = min(len(ego), len(kin), len(ghost), len(base), len(refi), F)
light_blue = np.array([250, 200, 120], np.float32)  # BGR sky blue
panels = [('Original human video', ''), ('Reference (kinematic)', 'final reference; light blue = original'),
          ('Baseline (physics)', 'recorded simulation'), ('Refined (physics)', 'recorded simulation')]
writer_path = TMP / 'tiled_raw.mp4'
wr = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (4 * PW, PH + HEAD))
for i in range(n):
    g = ghost[i].astype(np.float32)
    mask = (g.max(axis=2, keepdims=True) > 12).astype(np.float32) * .5
    k = kin[i].astype(np.float32) * (1 - mask) + mask * (.7 * light_blue + .3 * g)
    canvas = np.zeros((PH + HEAD, 4 * PW, 3), np.uint8); canvas[:HEAD] = (28, 28, 28)
    for j, im in enumerate((ego[i], k.clip(0, 255).astype(np.uint8), base[i], refi[i])):
        canvas[HEAD:, j * PW:(j + 1) * PW] = im
        title, sub = panels[j]
        cv2.putText(canvas, title, (j * PW + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'{i / fps:.2f}s  {sub}', (j * PW + 10, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (190, 190, 190), 1, cv2.LINE_AA)
    wr.write(canvas)
wr.release()
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(writer_path), '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT)], check=True)
(OUT.with_suffix('.json')).write_text(json.dumps(dict(root=str(ROOT), final_reference=str(FINAL), policy=str(POLICY),
    policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(), ego=str(EGO), frames=n, fps=fps,
    camera_position=cam.tolist(), camera_target=target.tolist(), refined_world=wr_, refined_world_aligned_cm=float(err_r[wr_]),
    refined_pass_fraction_3cm=float((err_r < 3).mean()), baseline_world=wb_, baseline_aligned_cm=float(err_b[wb_])), indent=2) + '\n')
print('DEMO', OUT, n, 'frames', flush=True)

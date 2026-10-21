"""Grasp stability under a growing disturbance force, per axis, rendered as one grid video (rows = grasps, columns = directions).
Each grasp: the reference plays until mid-hold (grasp forms in physics), then FREEZES; gravity is replaced by a force field along one
axis whose magnitude RAMPS from 0 to 3 m*g over 3 s (one world per direction: +x -x +y -y +z -z). The panel shows the force as a red
arrow scaled with its magnitude, the current force in units of m*g, and the in-hand motion; the moment the object slips (> 2 cm or 15 deg
relative to the palm) the panel turns red and freezes the 'breaking force'. Optional trained policy per row (deterministic).
Usage (training image, GPU): grasp_stability_video.py ROOT SEG_NPZ CAMERA.npz OUT.mp4 ROWS.json
  ROWS.json: [{"label": "...", "ref": "path.npz", "policy": null | "path.safetensors"}, ...]"""
import hashlib, json, os, re, subprocess, sys
from pathlib import Path
import cv2, imageio_ffmpeg, newton, numpy as np, warp as wp
import jax.numpy as jnp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.embodiments.binding import _frame_poses
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.runtime.jax import to_jax
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.visualization.viewer import ViewerConfig, configure_camera, configure_render_style, create_viewer
ROOT, SEG, CAM, OUT, ROWSF = (Path(x) for x in sys.argv[1:6]); ROWS = json.loads(ROWSF.read_text())
EP = ROOT.name; f = ROOT / 'frozen' / EP; TMP = OUT.parent / (OUT.stem + '_parts'); TMP.mkdir(parents=True, exist_ok=True)
DIRS = ['+x', '-x', '+y', '-y', '+z (up)', '-z (down)']; W = 6; PRE, RAMP, FMAX = 8, 60, 3.0
PW, PH, HEAD, LAB = 300, 240, 40, 170; FOV = 40.
B = ROOT; a = type('A', (), dict(phase='pilot', seed=44, worlds=1))(); train = ROOT / 'runs/demo_tmp'; total = interval = search_interval = W * 4096
cmd = re.search(r"cmd=\[(.*?)\]\n", (ROOT / 'code/run_mixed.py').read_text(), re.S).group(1)
overrides = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.', 'task.robot_reference', 'training.'))]
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    cfg = compose(config_name='train_flash_sac', overrides=overrides)
resolved = OmegaConf.to_container(cfg, resolve=True); apply_evaluation_composition(resolved, world_count=W)
env_cfg = instantiate_typed(OmegaConf.create(resolved['env']), RLEnvConfig); training = instantiate_typed(cfg.training, TrainingConfig)
m = np.load(SEG); key = sorted(k for k in m.files if k.endswith('_meta'))[0]; s0, e0, _kf, o = map(int, m[key]); side = key.split('_')[0]
ISO = os.environ.get('GS_ISOLATED') == '1'
# isolated grasp test (Dexonomy-style): no table, reset at the contact onset with the fingers still open, gravity OFF while the
# reference closes the fingers on the (floating) object; freeze once closed (+0.5 s settle), then ramp the force.
T_START = s0 if ISO else 0
t0 = (s0 + 20) if ISO else s0 + int(.6 * (e0 - s0)); PANEL = os.environ.get('GS_PANEL')
T_CLOSED = s0 + 9   # fingers fully closed (lead 0.1 s + close 0.3 s at 20 Hz); the reference is held here before the ramp
if ISO: PRE = t0 - T_START
def drop_table(model, solver=None):
    # isolated test: move every static shape (the support table) 10 m down so the object floats free
    lab = getattr(model, 'shape_label', None) or getattr(model, 'shape_key', None)
    m = np.array(['table' in str(x).lower() or 'support' in str(x).lower() for x in lab]) if lab is not None else (model.shape_body.numpy() == -1)
    tf = model.shape_transform.numpy(); tf[m, 2] -= 10.; model.shape_transform.assign(tf)   # the support table is its own body's shape
    print('[iso] moved', int(m.sum()), 'support shapes 10 m down', flush=True)
    if solver is not None: solver.notify_model_changed(newton.solvers.SolverNotifyFlags.SHAPE_PROPERTIES)
def mk(ref, worlds):
    return setup_scene(parquet=str(f / 'reference.parquet'), control_fps=env_cfg.sim.fps, motion_speed=1.0, embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                       collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=worlds, include_support=resolved['scene']['support'], decompose_objects=True,
                       robot_reference_path=str(ref), robot_reference_sha256=hashlib.sha256(Path(ref).read_bytes()).hexdigest())
camd = np.load(CAM); Twc = camd['world_from_rgb'].astype(np.float64)[min(t0, len(camd['world_from_rgb']) - 1)]; fwd = Twc[:3, 2] / np.linalg.norm(Twc[:3, 2])
def cam_for(p):
    return p - .30 * fwd + np.array([0, 0, .10]), p
with wp.ScopedDevice(os.environ.get('GS_DEVICE', 'cuda:0')):
    for r, row in enumerate(ROWS):
        if (TMP / f'row{r}.npz').exists(): continue
        setup = mk(row['ref'], W); s = setup.scene
        if ISO: drop_table(s.model)   # BEFORE the solver is built from the model (static geoms are baked in at solver creation)
        env = RLEnv(s, setup.reference, config=env_cfg)
        env.reset(frame_id=T_START); env.capture_step(); env.reset(frame_id=T_START); env.action_input.zero_()
        if ISO:
            s.model.gravity.assign(np.zeros_like(s.model.gravity.numpy())); env.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.MODEL_PROPERTIES)
        if row.get('policy'):
            adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
            obs = jnp.array(to_jax(env.reset(frame_id=T_START), (env.world_count, env.observation_dim)), copy=True)
            pst, pol, pk, _ = load_policy_for_inference(row['policy'], training, obs, adapter.action_dim, deterministic=True)
        ids = env.command.body_ids_w.numpy().reshape(W, -1)[:, o]; g0 = s.model.gravity.numpy().copy(); gn = float(np.linalg.norm(g0[0])) or 9.81
        unit = np.zeros((W, 3)); [unit.__setitem__((w, w // 2), 1. if w % 2 == 0 else -1.) for w in range(W)]
        ts = wp.array(np.full(W, T_CLOSED if ISO else t0, np.int32), dtype=wp.int32); frames, mags = [], []
        for t in range(T_START, t0 + RAMP):
            if t >= t0:
                mag = FMAX * (t - t0 + 1) / RAMP; s.model.gravity.assign((unit * mag * gn).astype(np.float32))
                env.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.MODEL_PROPERTIES); wp.copy(env.timestep, ts)
            else:
                mag = 0.
                if ISO and t >= T_CLOSED: wp.copy(env.timestep, ts)   # hold the closed grasp while it settles (gravity off)
            if row.get('policy'):
                act, pk = pol(pst, obs, pk); obs = jnp.array(adapter.step(act).observation, copy=True)
            else: env.step()
            if t >= max(T_START, t0 - PRE): frames.append(env.state_0.body_q.numpy().copy()); mags.append(mag)
        bq = np.asarray(frames); nb = bq.shape[1] // W; B4 = bq.reshape(len(bq), W, nb, 7)
        h = [hh.side for hh in s.layout.hands].index(side); palms = tuple(hh.palm_frame for hh in s.layout.hands); slip, rot = [], []
        for w in range(W):
            pp, pq = _frame_poses(B4[:, w], palms); ob = B4[:, w, ids[w] - w * nb]; Rp = Rotation.from_quat(pq[:, h])
            rel = Rp.inv().apply(ob[:, :3] - pp[:, h]); rr = Rp.inv() * Rotation.from_quat(ob[:, 3:])
            slip.append(np.linalg.norm(rel - rel[PRE], axis=1) * 100); rot.append(np.degrees((rr * rr[PRE].inv()).magnitude()))
        slip, rot = np.asarray(slip), np.asarray(rot); broken = ((slip >= 2.) | (rot >= 15.)); broken[:, :PRE + 1] = False   # only after the freeze
        brk = [float(np.asarray(mags)[np.argmax(broken[w])]) if broken[w].any() else None for w in range(W)]
        held0 = bool(np.linalg.norm(B4[PRE, 0, ids[0], :3] - B4[PRE, 0, ids[0], :3]) == 0)
        np.savez(TMP / f'row{r}.npz', bq=bq, nb=nb, ids=ids, slip=slip, rot=rot, mags=np.asarray(mags), unit=unit,
                 obj=np.stack([B4[:, w, ids[w] - w * nb, :3] for w in range(W)], 1), brk=np.array([-1. if b is None else b for b in brk]))
        print('ROW', r, row['label'], 'breaking force (x m*g) per direction:', dict(zip(DIRS, brk)), flush=True)
    if PANEL is not None:
        r, w = map(int, PANEL.split(':')); d = np.load(TMP / f'row{r}.npz'); bq, nb = d['bq'], int(d['nb'])
        single = mk(ROWS[r]['ref'], 1).scene
        if ISO: drop_table(single.model)
        pos, tgt = cam_for(d['obj'][PRE, w].astype(np.float64))
        vc = ViewerConfig(backend='mp4', output_path=str(TMP / f'r{r}w{w}.mp4'), num_frames=len(bq), video_fps=10, width=PW, height=PH, headless=True, realtime=False,
                          camera_position=tuple(pos), camera_target=tuple(tgt), background_color=(.08, .08, .1), support_color=(.96, .62, .33))
        v = create_viewer(vc); v.set_model(single.model); configure_camera(v, vc); configure_render_style(v, vc, scene=single); v.camera.fov = FOV
        st = single.model.state()
        try:
            for i in range(len(bq)):
                st.body_q.assign(bq[i].reshape(W, nb, 7)[w].astype(np.float32)); v.begin_frame(i / 10); v.log_state(st); v.end_frame()
        finally:
            v.close()
        sys.exit(0)
for r in range(len(ROWS)):
    for w in range(W):
        if not (TMP / f'r{r}w{w}.mp4').exists():
            subprocess.run([sys.executable, __file__, *sys.argv[1:]], check=True, env={**os.environ, 'GS_PANEL': f'{r}:{w}'})
def frames_of(p):
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, im = cap.read()
        if not ok: break
        out.append(im)
    return out
def project(X, pos, tgt):
    fz = (tgt - pos) / np.linalg.norm(tgt - pos); rx = np.cross(fz, [0, 0, 1.]); rx /= np.linalg.norm(rx); uy = np.cross(rx, fz)
    v = X - pos; z = max(v @ fz, 1e-3); fp = (PH / 2) / np.tan(np.radians(FOV) / 2)
    return int(PW / 2 + fp * (v @ rx) / z), int(PH / 2 - fp * (v @ uy) / z)
def _fix(d):
    d = dict(d); br = ((d['slip'] >= 2.) | (d['rot'] >= 15.)); br[:, :PRE + 1] = False
    d['brk'] = np.array([float(d['mags'][np.argmax(br[w])]) if br[w].any() else -1. for w in range(W)]); return d
D = [_fix(np.load(TMP / f'row{r}.npz')) for r in range(len(ROWS))]; V = [[frames_of(TMP / f'r{r}w{w}.mp4') for w in range(W)] for r in range(len(ROWS))]
n = min(len(x) for row in V for x in row); Wd, Hd = LAB + W * PW, HEAD + len(ROWS) * PH
raw = TMP / 'grid.mp4'; wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), 10, (Wd, Hd))
for i in range(n):
    cv = np.full((Hd, Wd, 3), 24, np.uint8)
    mag = float(D[0]['mags'][i])
    cv2.putText(cv, ('Isolated grasp test: grasp -> settle -> force ramps 0 -> 3 m*g along one axis per column' if ISO else 'Grasp stability: force ramps 0 -> 3 m*g along one axis per column (hand frozen at mid-hold)'), (10, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
    _ph = (f'applied force now: {mag:.2f} m*g' if i >= PRE else ('1) hand placed at the grasp, fingers closing on the spoon (no table, gravity off)' if ISO and i < T_CLOSED - T_START
           else ('2) grasp formed, settling' if ISO else 'grasp forming (before freeze)')))
    cv2.putText(cv, _ph if i < PRE else '3) ' + _ph, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, .5, (90, 90, 255) if i >= PRE else (200, 200, 200), 1, cv2.LINE_AA)
    for w in range(W):
        cv2.putText(cv, DIRS[w], (LAB + w * PW + PW // 2 - 30, HEAD - 4), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
    for r, row in enumerate(ROWS):
        y = HEAD + r * PH
        for k, line in enumerate(row['label'].split('|')):
            cv2.putText(cv, line, (8, y + 30 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
        brk = D[r]['brk']; held = int(sum(b < 0 for b in brk))
        cv2.putText(cv, f'held to 3 m*g: {held}/6', (8, y + PH - 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (120, 220, 120) if held >= 4 else (90, 90, 240), 1, cv2.LINE_AA)
        for w in range(W):
            im = V[r][w][i].copy(); d = D[r]; j = min(i, d['slip'].shape[1] - 1); pos, tgt = cam_for(d['obj'][PRE, w].astype(np.float64))
            lost_at = d['brk'][w]; lost = lost_at >= 0 and i >= PRE and float(d['mags'][j]) >= lost_at
            if i >= PRE:
                P0 = d['obj'][j, w].astype(np.float64); L = .03 + .05 * float(d['mags'][j]) / FMAX
                a0, a1 = project(P0, pos, tgt), project(P0 + L * d['unit'][w], pos, tgt)
                cv2.arrowedLine(im, a0, a1, (40, 40, 255), 4, cv2.LINE_AA, tipLength=.3)
            col = (60, 60, 230) if lost else ((80, 200, 80) if i >= PRE else (120, 120, 120))
            cv2.rectangle(im, (1, 1), (PW - 2, PH - 2), col, 3)
            txt = (f'LOST at {lost_at:.2f} m*g' if lost else f'in hand {d["slip"][w, j]:.1f} cm / {d["rot"][w, j]:.0f} deg')
            cv2.putText(im, txt, (8, PH - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1, cv2.LINE_AA)
            cv[y:y + PH, LAB + w * PW:LAB + (w + 1) * PW] = im
    wr.write(cv)
    if i == n - 1:
        for _ in range(20): wr.write(cv)   # hold the final frame 2 s
wr.release()
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(OUT)], check=True)
summary = {ROWS[r]['label']: {DIRS[w]: (None if D[r]['brk'][w] < 0 else round(float(D[r]['brk'][w]), 2)) for w in range(W)} for r in range(len(ROWS))}
OUT.with_suffix('.json').write_text(json.dumps(summary, indent=1)); print('GRASP VIDEO', OUT, json.dumps(summary), flush=True)

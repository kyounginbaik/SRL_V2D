"""Upper level from scratch, all in Newton: find a hand reference for a given object path with no demo hand at all.

Candidate (13 numbers) = a rigid grip: pen pose in the hand frame (3 offset from the hand's tripod point + 3 rotation vector),
grip point along the pen axis (1), flexion per finger as a fraction of its range (5), thumb opposition CMC_AA (1).
The hand reference is the object path composed with the inverse grip, fingers constant. The tripod point (mean of the
thumb/index/middle distal links at half closure) comes from Newton forward kinematics of the hand.
Scoring: K candidates are tiled in time (PIB_SEG/PIB_PROBE), every world starts at a slot's first frame with no assistance,
rolls out with zero policy residual (pure PD tracking), friction randomized per world (FRIC_DR). Held = pen within 3 cm of its
designed place in the hand. Fitness = good writing frames / writing frames + 0.5 * held frames / CLIP.
Search: gen 0 uniform random pen orientations; then 8 parents (best mean fitness so far) x 16 children each.
Usage (container): scratch.py CKPT_FOR_CONFIG TILED_PARQUET SINGLE_PARQUET OUT_DIR [GENS]
"""
import json, os, sys, time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import newton
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/workspace/src")
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.retargeting.manipulation_bilevel import LocalProjector
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.evaluation import harness as H

ckpt, tiled_pq, single_pq, out = sys.argv[1:5]
GENS = int(sys.argv[5]) if len(sys.argv) > 5 else 30
SEG, CLIP, LOST, NPAR = 472, 450, 0.03, 8
out = Path(out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))

cfg_d = checkpoint_config(read_checkpoint_metadata(Path(ckpt)))
cfg_d["task"]["robot_reference_path"] = None; cfg_d["task"]["robot_reference_sha256"] = None
for tb in H._termination_blocks(cfg_d):  # no failure terminations: held/dropped is measured here
    if H._is_recon_body_termination(tb):
        for nm in H._RECON_BODY_THRESHOLD_FIELDS: tb[nm] = H._DISABLED_THRESHOLD
    else:
        for nm in H._FAILURE_TERMS:
            if nm in tb: tb[nm]["enabled"] = False
cfg_d["env"]["voc_scale"] = 0.0  # training-style start: VOC only during the post-reset settle (PIB_TR=1), then none
cfg_d["scene"]["world_count"] = int(os.environ.get("WORLDS", "2048"))
cfg = OmegaConf.create(cfg_d)
env_config = instantiate_typed(cfg.env, RLEnvConfig)
embodiment = instantiate_typed(cfg.embodiment, Embodiment)
collision = instantiate_typed(cfg.collision, CollisionPolicy)


def make_setup(parquet, worlds):
    return setup_scene(parquet=parquet, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=embodiment,
                       collision=collision, world_count=worlds, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
                       contact_friction=cfg.scene.get("contact_friction", 1.0), object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0))


tiled = make_setup(tiled_pq, cfg.scene.world_count)
single = make_setup(single_pq, 1)
TR = tiled.scene.robot_reference
K = TR.num_frames // SEG
proj = LocalProjector(embodiment, TR)
hand = next(h for h in proj.layout.hands if h.side == "right")
lo, hi = proj.model.joint_limit_lower.numpy(), proj.model.joint_limit_upper.numpy()
fdof = list(hand.finger_dof_ids); fids = list(hand.finger_q_ids)
names = list(tiled.reference.finger_joint_names("right"))
FLEX = {f: [i for i, nm in enumerate(names) if f"_{f}_" in nm and any(x in nm for x in ("FE", "PIP", "DIP", "IP"))] for f in ("thumb", "index", "middle", "ring", "pinky")}
TAA = names.index("right_thumb_CMC_AA")
flo, fhi = lo[fdof], hi[fdof]


def fingers(c, taa):
    f = np.clip(np.zeros(len(names)), flo, fhi)
    for k, fn in enumerate(FLEX):
        for i in FLEX[fn]:
            f[i] = flo[i] + np.clip(c[k], 0, 1) * (fhi[i] - flo[i])
    f[TAA] = np.clip(taa, flo[TAA], fhi[TAA])
    return f


# tripod point from Newton FK: hand root at identity, half closure
sm = single.scene.model; jq = sm.joint_q.numpy().copy(); st = sm.state()
jq[list(hand.wrist_pos_q_ids)] = 0.0; jq[hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4] = [0, 0, 0, 1]
jq[fids] = fingers(np.full(5, 0.5), 0.0)
sm.joint_q.assign(jq); newton.eval_fk(sm, sm.joint_q, sm.joint_qd, st)
blab = list(getattr(sm, "body_label", None) or getattr(sm, "body_key"))
bq = st.body_q.numpy()
TRIPOD = np.mean([bq[blab.index(f"right_sharpa_wave/right_{f}_DP"), :3] for f in ("thumb", "index", "middle")], 0)
print(f"[scratch] K={K} worlds={cfg.scene.world_count} tripod(hand frame)={np.round(TRIPOD, 3)}", flush=True)

tiled_target = np.asarray(TR.joint_target, float)
base_q = np.asarray(single.scene.robot_reference.joint_q, float)[:CLIP]
base_q = np.concatenate([base_q, np.repeat(base_q[-1:], SEG - CLIP, 0)])
gp = np.asarray(tiled.reference.object_body_pos_w(), float)[:SEG, 0]
gq = np.asarray(tiled.reference.object_body_quat_w(), float)[:SEG, 0]
gr = R.from_quat(gq[:, [1, 2, 3, 0]])
board = json.loads(Path("/oak/write/board.json").read_text())
tip_local, normal, surface = np.array(board["tip_local"]), np.array(board["normal"]), np.array(board["surface_point"])
gtip = gr.apply(tip_local) + gp
hgt = (gtip - surface) @ normal
on = hgt < np.percentile(hgt[:CLIP], 20) + 0.002; on[CLIP:] = False
idx = np.flatnonzero(on); on[:idx[0]] = False; on[idx[-1] + 1:] = False
LOWS = np.r_[[-np.inf] * 6, -0.05, [0.0] * 5, -0.35]; HIGHS = np.r_[[np.inf] * 6, 0.06, [1.0] * 5, 0.35]
SIG0 = np.r_[[0.01] * 3, [0.25] * 3, 0.01, [0.08] * 5, 0.08]


def grip(x):
    """pen pose in the hand frame: rotation, and position of the pen origin"""
    Rhp = R.from_rotvec(x[3:6])
    return Rhp, TRIPOD + x[:3] - Rhp.apply([0, 0, x[6]])


def candidate(x):
    Rhp, thp = grip(x)
    q = base_q.copy()
    Rw = gr * Rhp.inv()
    q[:, hand.wrist_pos_q_ids] = gp - Rw.apply(thp)
    q[:, hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4] = Rw.as_quat()
    q[:, fids] = fingers(x[7:12], x[12])
    q[CLIP:] = q[CLIP - 1]
    return q


def bind(q, target, fps):
    target = target.copy()
    target[:, hand.wrist_pos_dof_ids] = q[:, hand.wrist_pos_q_ids]
    target[:, hand.wrist_orient_dof_ids] = R.from_quat(q[:, hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4]).as_rotvec()
    target[:, hand.finger_dof_ids] = q[:, hand.finger_q_ids]
    return build_robot_reference_binding(proj.builder, proj.layout, q.astype(np.float32), target.astype(np.float32), fps)


def evaluate(xs):
    qs = [candidate(x) for x in xs]
    q = np.concatenate(qs + [np.repeat(qs[-1][-1:], TR.num_frames - K * SEG, 0)], 0)
    env = RLEnv(replace(tiled.scene, robot_reference=bind(q, tiled_target, TR.fps)), tiled.reference, config=deepcopy(env_config))
    env.reset()
    W = env.world_count
    t0 = env.timestep.numpy().copy(); slot = t0 // SEG
    assert np.all(t0 % SEG == 0), np.unique(t0 % SEG)[:10]
    design = np.stack([grip(xs[k])[1] for k in slot])
    started = np.zeros(W, bool); rel0 = np.zeros((W, 3)); alive = np.ones(W, bool); held = np.zeros(W); good = np.zeros(W)
    ids = env.command.body_ids_w.numpy().reshape(W, -1)[:, 0]
    for f in range(CLIP - 1 + 40):
        env.action_input.zero_(); env.step(None)
        t = env.timestep.numpy(); bq = env.state_0.body_q.numpy()[ids]
        jq = env.state_0.joint_q.numpy().reshape(W, -1)
        wr = R.from_quat(jq[:, hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4])
        rel = wr.inv().apply(bq[:, :3] - jq[:, hand.wrist_pos_q_ids])
        fr = t - t0
        new = ~started & (fr > 0); rel0[new] = rel[new]; started |= new  # grip as formed after the settle
        act = started & (fr >= 1) & (fr <= CLIP - 1)
        alive &= ~act | (np.linalg.norm(rel - rel0, axis=1) < LOST)
        frc = np.clip(fr, 0, SEG - 1)
        te = np.linalg.norm(R.from_quat(bq[:, 3:7]).apply(tip_local) + bq[:, :3] - gtip[frc], axis=1)
        held += alive & act
        good += alive & act & on[frc] & (te < LOST)
        # pen must also stay near the designed place right after the settle (formed grip close to the plan)
        if (fr == 1).any(): alive &= ~(new & (np.linalg.norm(rel - design, axis=1) > LOST))
    fit = good / on.sum() + 0.5 * held / (CLIP - 1)
    res = [dict(fit=float(fit[slot == k].mean()), held=float((held[slot == k] >= CLIP - 2).mean()), kept=float(held[slot == k].mean() / (CLIP - 1)),
                good=float(good[slot == k].mean() / on.sum()), n=int((slot == k).sum())) for k in range(K)]
    return res, qs


def perturb(x, s):
    y = x.copy()
    y[:3] += s[:3] * rng.standard_normal(3)
    y[3:6] = (R.from_rotvec(x[3:6]) * R.from_rotvec(s[3:6] * rng.standard_normal(3))).as_rotvec()
    y[6:] += s[6:] * rng.standard_normal(7)
    return np.clip(y, LOWS, HIGHS)


archive = {}  # key -> dict(x, fits)
if os.environ.get("X0", "null") != "null":  # resume/export: seed the archive with a given grip
    j = json.loads(Path(os.environ["X0"]).read_text()); x0 = np.array(j["x"], float)
    archive[tuple(np.round(x0, 9))] = dict(x=x0, fits=[float(j.get("fit", 0.0))], res=j.get("res", {}))
log = (out / "scratch_tr.jsonl").open("a")
sig = SIG0.copy()
for g in range(GENS):
    t1 = time.time()
    if g == 0:
        xs = [np.r_[0.01 * rng.standard_normal(3), R.random(random_state=rng.integers(1 << 31)).as_rotvec(), rng.uniform(0, 0.05),
                    rng.uniform(0.25, 0.7, 5), rng.uniform(-0.3, 0.3)] for _ in range(K)]
    else:
        par = sorted(archive.values(), key=lambda a: -np.mean(a["fits"]))[:NPAR]
        per = K // len(par)
        xs = []
        for p in par:
            xs.append(p["x"].copy())  # re-measure the parent
            xs += [perturb(p["x"], sig) for _ in range(per - 1)]
        xs = (xs + [perturb(par[0]["x"], sig) for _ in range(K - len(xs))])[:K]
    res, qs = evaluate(xs)
    for x, r in zip(xs, res):
        if r["n"] == 0: continue
        key = tuple(np.round(x, 9))
        archive.setdefault(key, dict(x=x, fits=[], res=r))["fits"].append(r["fit"]); archive[key]["res"] = r
    top = sorted(archive.values(), key=lambda a: -np.mean(a["fits"]))[:NPAR]
    if g >= 8: sig = np.maximum(sig * 0.93, SIG0 * 0.15)
    f = np.array([r["fit"] if r["n"] else -1.0 for r in res])
    rec = dict(gen=g, gen_best=res[int(f.argmax())], gen_mean_fit=float(f.mean()), gen_held_any=int(sum(r["held"] > 0 for r in res)),
               top=[dict(fit=float(np.mean(a["fits"])), n=len(a["fits"]), res=a["res"], x=np.round(a["x"], 4).tolist()) for a in top[:3]], sec=time.time() - t1)
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[scratch] gen {g} mean {f.mean():.3f} gen-best {res[int(f.argmax())]} | top {np.mean(top[0]['fits']):.3f} (n={len(top[0]['fits'])}) {top[0]['res']} | {time.time() - t1:.0f}s", flush=True)

# export the best (by mean over its re-measurements) as an ordinary single-clip reference snapshot
best = sorted(archive.values(), key=lambda a: -np.mean(a["fits"]))[0]
ST = single.scene.robot_reference
b = bind(candidate(best["x"])[:CLIP], np.asarray(ST.joint_target, float), ST.fps)
path = out / "best.npz"
sha = save_snapshot(path, b, single.scene.layout, provenance=dict(mode="newton_scratch_grip", source_parquet=single_pq,
                                                                 object_contact_sha256=command_digest(single.reference)))
(out / "best.json").write_text(json.dumps(dict(path=str(path), sha256=sha, fit=float(np.mean(best["fits"])), n=len(best["fits"]), x=list(map(float, best["x"])), res=best["res"])))
print("[scratch] wrote", path, sha, flush=True)
print("SCRATCH_DONE", flush=True)

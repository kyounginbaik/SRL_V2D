"""Open-loop practice in the training simulator (UniRetarget upper level, no IK, no policy).

Each generation tiles K candidate references in time (PIB_SEG frames each; env PIB_SEG and PIB_PROBE=1 make every
world start at a random slot's first frame with no assistance) and rolls all worlds out with zero policy residual,
i.e. the PD hand simply follows the candidate. Worlds have randomized friction (FRIC_DR), so each candidate gets
~W/K physical trials at once. Candidate = parent hand motion + a constant grip edit in the object frame
(3 translation + 3 rotation) + one squeeze per finger (5) = 11 numbers. Elitist (1+15) search over the 11 numbers.
Held = the pen has not moved 3 cm relative to the hand. Fitness = mean over trials of (writing frames with the tip within 3 cm) / (writing frames) + 0.5 * (frames held) / CLIP.
Usage (in the container): oloop.py CKPT_FOR_CONFIG TILED_PARQUET PARENT_PARQUET OUT_DIR [GENERATIONS]
"""
import json, sys, time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
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

ckpt, tiled_pq, parent_pq, out = sys.argv[1:5]
DEBUG = bool(int(__import__('os').environ.get('DEBUG', '0')))
GENS = int(sys.argv[5]) if len(sys.argv) > 5 else 25
SEG, CLIP, LOST = 472, 450, 0.03
out = Path(out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

cfg_d = checkpoint_config(read_checkpoint_metadata(Path(ckpt)))
cfg_d["task"]["robot_reference_path"] = None; cfg_d["task"]["robot_reference_sha256"] = None
# no failure terminations: a dropped pen keeps rolling out, and "held" is measured here (pen within 3 cm of its goal)
from flash_chord.evaluation import harness as H
for tb in H._termination_blocks(cfg_d):
    if H._is_recon_body_termination(tb):
        for nm in H._RECON_BODY_THRESHOLD_FIELDS: tb[nm] = H._DISABLED_THRESHOLD
    else:
        for nm in H._FAILURE_TERMS:
            if nm in tb: tb[nm]["enabled"] = False
cfg_d["env"]["voc_scale"] = 0.0; cfg_d["env"]["voc"]["max_force"] = 0.0; cfg_d["env"]["voc"]["max_torque"] = 0.0  # no assistance, exactly as in evaluation
cfg_d["scene"]["world_count"] = int(__import__("os").environ.get("WORLDS", "2048"))
cfg = OmegaConf.create(cfg_d)
env_config = instantiate_typed(cfg.env, RLEnvConfig)
embodiment = instantiate_typed(cfg.embodiment, Embodiment)
collision = instantiate_typed(cfg.collision, CollisionPolicy)


def make_setup(parquet, worlds):
    return setup_scene(parquet=parquet, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=embodiment,
                       collision=collision, world_count=worlds, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
                       contact_friction=cfg.scene.get("contact_friction", 1.0), object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0))


tiled = make_setup(tiled_pq, cfg.scene.world_count)
single = make_setup(parent_pq, 1)
TR = tiled.scene.robot_reference
K = TR.num_frames // SEG
proj = LocalProjector(embodiment, TR)
hand = next(h for h in proj.layout.hands if h.side == "right")
lo, hi = proj.model.joint_limit_lower.numpy(), proj.model.joint_limit_upper.numpy()
flim = np.stack([lo[list(hand.finger_dof_ids)], hi[list(hand.finger_dof_ids)]], 1)
names = list(tiled.reference.finger_joint_names("right")); fids = list(hand.finger_q_ids)
groups = [[fids[i] for i, nm in enumerate(names) if f"_{f}_" in nm and any(x in nm for x in ("FE", "PIP", "DIP", "IP"))]
          for f in ("thumb", "index", "middle", "ring", "pinky")]
tiled_target = np.asarray(TR.joint_target, float)
parent = np.asarray(single.scene.robot_reference.joint_q, float)[:CLIP]
parent = np.concatenate([parent, np.repeat(parent[-1:], SEG - CLIP, 0)])
gp = np.asarray(tiled.reference.object_body_pos_w(), float)[:SEG, 0]
gq = np.asarray(tiled.reference.object_body_quat_w(), float)[:SEG, 0]
gr = R.from_quat(gq[:, [1, 2, 3, 0]])
board = json.loads(Path("/oak/write/board.json").read_text())
tip_local, normal, surface = np.array(board["tip_local"]), np.array(board["normal"]), np.array(board["surface_point"])
gtip = gr.apply(tip_local) + gp
h = (gtip - surface) @ normal
on = h < np.percentile(h[:CLIP], 20) + 0.002; on[CLIP:] = False
idx = np.flatnonzero(on); on[:idx[0]] = False; on[idx[-1] + 1:] = False
print(f"[oloop] K={K} worlds={cfg.scene.world_count} writing frames={on.sum()}", flush=True)
SIG0 = np.r_[[0.006] * 3, [np.radians(8)] * 3, [0.08] * 5]


def candidate(x):
    """Parent motion + constant grip edit in the object frame + per-finger squeeze."""
    dp, dr, sq = x[:3], x[3:6], x[6:]
    q = parent.copy()
    cp = q[:, hand.wrist_pos_q_ids]; cr = R.from_quat(q[:, hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4])
    p = gp + gr.apply(R.from_rotvec(dr).apply(gr.inv().apply(cp - gp)) + dp)
    r = gr * R.from_rotvec(dr) * gr.inv() * cr
    q[:, hand.wrist_pos_q_ids] = p; q[:, hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4] = r.as_quat()
    for g, s in zip(groups, sq):
        q[:, g] += s
    q[:, fids] = np.clip(q[:, fids], flim[:, 0], flim[:, 1])
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
    q_ref = q
    t0 = env.timestep.numpy().copy(); slot = t0 // SEG
    assert np.all(t0 % SEG == 0), np.unique(t0 % SEG)[:10]
    alive = np.ones(W, bool); held = np.zeros(W); good = np.zeros(W); prev = t0
    ids = env.command.body_ids_w.numpy().reshape(W, -1)[:, 0]
    for f in range(CLIP - 1):
        env.action_input.zero_(); env.step(None)
        t = env.timestep.numpy()
        bq = env.state_0.body_q.numpy()[ids]
        if DEBUG and f < 30:
            jq = env.state_0.joint_q.numpy().reshape(W, -1)[:, hand.wrist_pos_q_ids]
            print("[dbg] f", f, "t", t[:3], "pen err cm", np.round(np.median(np.linalg.norm(bq[:, :3] - gp[f + 1], axis=1)) * 100, 2),
                  "pen", np.round(bq[0, :3] * 100, 1), "goal", np.round(gp[f + 1] * 100, 1), "q0:3", np.round(jq[0] * 100, 1), "ref", np.round(q_ref[t[0], hand.wrist_pos_q_ids] * 100, 1), "fing", np.round(env.state_0.joint_q.numpy().reshape(W, -1)[0, fids[:4]], 2), np.round(q_ref[t[0], fids[:4]], 2), flush=True)
        jq = env.state_0.joint_q.numpy().reshape(W, -1)
        wr = R.from_quat(jq[:, hand.wrist_orient_q_id: hand.wrist_orient_q_id + 4])
        rel = wr.inv().apply(bq[:, :3] - jq[:, hand.wrist_pos_q_ids])  # pen in the hand frame
        if f == 0:
            rel0 = rel.copy()
        alive &= (t == prev + 1) & (np.linalg.norm(rel - rel0, axis=1) < LOST); prev = t.copy()  # held = pen has not slipped 3 cm in the hand
        te = np.linalg.norm(R.from_quat(bq[:, 3:7]).apply(tip_local) + bq[:, :3] - gtip[f + 1], axis=1)
        held += alive
        if on[f + 1]:
            good += alive & (te < LOST)
    fit = good / on.sum() + 0.5 * held / (CLIP - 1)
    res = [dict(fit=float(fit[slot == k].mean()), held=float((held[slot == k] >= CLIP - 2).mean()), kept=float(held[slot == k].mean() / (CLIP - 1)),
                good=float(good[slot == k].mean() / on.sum()), n=int((slot == k).sum())) for k in range(K)]
    return res, qs


mu, sig = (np.array(json.loads(Path(__import__("os").environ["X0"]).read_text())["x"], float) if __import__("os").environ.get("X0", "null") != "null" else np.zeros(11)), SIG0.copy()
best = (-1, None, None, None)
log = (out / "oloop.jsonl").open("a")
for g in range(GENS):
    t1 = time.time()
    xs = [mu.copy()] + [mu + sig * rng.standard_normal(11) for _ in range(K - 1)]
    res, qs = evaluate(xs)
    f = np.array([r["fit"] for r in res]); order = np.argsort(-f)
    if f[order[0]] > best[0]:
        best = (float(f[order[0]]), xs[order[0]], qs[order[0]], res[order[0]])
    # elitist (1+lambda) search: always perturb the best candidate seen so far (CEM averaging lost the narrow ridge)
    improved = best[1] is not None and np.allclose(best[1], xs[order[0]]) and f[order[0]] >= best[0] - 1e-12
    mu = np.array(best[1], float); sig = sig if improved else np.maximum(sig * 0.85, SIG0 * 0.1)
    rec = dict(gen=g, mean_fit=res[0], best_fit=res[order[0]], best_x=np.round(xs[order[0]], 4).tolist(), mu=np.round(mu, 4).tolist(), sec=time.time() - t1,
               all_fit=np.round(f, 3).tolist())
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[oloop] gen {g} mean-candidate {res[0]} | best {res[order[0]]} | {time.time() - t1:.0f}s", flush=True)

# export the best as an ordinary single-clip reference snapshot (+ the mean)
fit, x, q, r = best
ST = single.scene.robot_reference
for name, qq in (("best", q), ("mean", candidate(mu))):
    b = bind(qq[:CLIP], np.asarray(ST.joint_target, float), ST.fps)
    path = out / f"{name}.npz"
    sha = save_snapshot(path, b, single.scene.layout, provenance=dict(mode="open_loop_practice", source_parquet=parent_pq,
                                                                     object_contact_sha256=command_digest(single.reference)))
    (out / f"{name}.json").write_text(json.dumps(dict(path=str(path), sha256=sha, fit=fit, x=list(map(float, x)), res=r)))
    print("[oloop] wrote", path, sha, flush=True)
print("OLOOP_DONE", flush=True)

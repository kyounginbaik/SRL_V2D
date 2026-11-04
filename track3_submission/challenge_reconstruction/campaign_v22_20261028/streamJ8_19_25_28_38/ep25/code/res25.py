"""Stream J8 (v23 ep25): physical fine-tuning of a distilled actor's PHASE WEIGHTS on its own closed-loop rollout
(port of streamH_ep32/code/actor_res.py to the two-handed board episode).

Architecture unchanged (plain LearnedPolicy via ComponentPolicy): output = RBF(t) @ W + state @ F. A time-local residual delta(t) is
baked into W: W'[j,a] = W[j,a] + delta_a(t_j)/scale_a. Feedback weights, scene, start state untouched. Every evaluation is a full
closed-loop rollout with the same controller object / substep schedule as evaluate_policy_component, scored with the evaluator's own
MP-SR proxy (evaluate_mp_sr: anchors, initial-scene alignment) vs the world reference (= target_final_v5 in our body frames; README E:
proxy == score_any here).

residual table (T x 16): left wrist pos xyz, rot xyz | right wrist pos xyz, rot xyz | L flex sq, L thumb sq, R flex sq, R thumb sq.
left hand <-> wooden_piece_1, right hand <-> wooden_piece_2.
modes: eval | ilc (rigid-hold wrist correction per side in its window) | cem (knot residual) | save
usage: res25.py MODE --actor DIR --out DIR [--lw t0,t1] [--rw t0,t1] [--table npz] ..."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR, hand as Hh
import evaluate_mp_sr as EM
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
NCH = 16
OBJ = ('wooden_piece_1', 'wooden_piece_2'); SIDES = ('left', 'right')


class Env:
    def __init__(self, cfg):
        self.cfg = cfg; actor = Path(cfg['actor']); ev = json.loads((actor / 'evaluation/result.json').read_text())
        self.scene = ev['execution']['scene']; self.epd = Path(ev['input_dir']); self.ep = 25
        self.m = mujoco.MjModel.from_xml_path(self.scene); self.dt = float(self.m.opt.timestep)
        from component_policy import ComponentPolicy
        self.ckpt = cfg.get('ckpt') or str(actor / 'training/policy.npz'); self.pol = ComponentPolicy(self.ckpt); self.d = self.pol.data
        self.W0 = self.d['phase_weights'].copy(); self.fps = float(self.d['fps']); self.T = int(self.d['frames']); self.dur = float(self.d['duration_s'])
        self.t = np.arange(self.T) / self.fps; self.tc = self.d['centers'] * self.dur
        names = list(self.d['actuator_names']); self.scale = self.d['action_scale']
        J = ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')
        self.wa = {s: [names.index(f'robot_{s}_{s}_{j}_position') for j in J] for s in SIDES}
        self.wq = {s: [int(self.m.joint(f'robot_{s}_{s}_{j}').qposadr[0]) for j in J] for s in SIDES}
        self.flex = []
        for si, s in enumerate(SIDES):
            rig = Hh.HandRig.build(self.m, f'robot_{s}_', s)
            for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                for jn in rig.flexion_joints(fi):
                    jid = self.m.joint(f'robot_{s}_{s}_' + jn).id if isinstance(jn, str) else int(jn)
                    hits = np.flatnonzero(self.m.actuator_trnid[:, 0] == jid)
                    if len(hits) == 1: self.flex.append((12 + 2 * si + (1 if fi == 'thumb' else 0), int(hits[0])))
        self.oa = [int(self.m.joint(o + '_free').qposadr[0]) for o in OBJ]
        man = json.loads((self.epd / 'input_manifest.json').read_text())
        self.ref = EM.load_reference(man['source']); self.S = np.asarray(man['source_from_setup'], float)
        geo = json.loads((self.epd / 'metric_geometry.json').read_text())
        self.anchors, self.axes, self.kinds = EM.validate_geometry(self.ref['object_names'], geo)
        assert list(self.ref['object_names']) == list(OBJ)
        r = dict(self.ref); r['object_positions'] = r['object_positions'] + np.einsum('twoij,oj->twoi', EM.rotation_matrices(r['object_quaternions_wxyz']), self.anchors)
        self.refa = r
        self.qt = TR.load_reference(self.epd / 'tracking_reference.npz', self.m)['qpos']
        self.table0 = np.zeros((self.T, NCH))
        if cfg.get('table'): self.table0 = np.load(cfg['table'])['table']

    def weights(self, table):
        W = self.W0.copy(); tv = np.stack([np.interp(self.tc, self.t, table[:, c]) for c in range(NCH)], 1)
        for si, s in enumerate(SIDES):
            for i, a in enumerate(self.wa[s]): W[:, a] += tv[:, 6 * si + i] / self.scale[a]
        for c, a in self.flex: W[:, a] += tv[:, c] / self.scale[a]
        return W

    def rollout(self, table):
        self.d['phase_weights'] = self.weights(table); m = self.m; pol = self.pol; dt = self.dt
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        Q = np.empty((self.T, m.nq)); Q[0] = d.qpos
        for k in range(1, self.T):
            n = round(k / (self.fps * dt)) - round((k - 1) / (self.fps * dt))
            for _ in range(n):
                d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
            Q[k] = d.qpos
        return Q

    def per_frame(self, Q):
        pos = np.stack([Q[:, a:a + 3] for a in self.oa], 1)[:, None]; quat = np.stack([Q[:, a + 3:a + 7] for a in self.oa], 1)[:, None]
        tr = dict(timestamps_seconds=self.t, object_positions=pos, object_quaternions_wxyz=quat, object_names=list(OBJ))
        tr = EM.apply_world_transform(tr, self.S)
        tr['object_positions'] = tr['object_positions'] + np.einsum('twoij,oj->twoi', EM.rotation_matrices(tr['object_quaternions_wxyz']), self.anchors)
        al, _ = EM.align_initial_scene(tr, self.refa, self.axes, self.kinds)
        p = np.linalg.norm(al['object_positions'] - self.refa['object_positions'], axis=-1)[:, 0] * 100
        a = np.degrees(EM.angular_errors(al['object_quaternions_wxyz'], self.refa['object_quaternions_wxyz'], self.axes, self.kinds))[:, 0]
        return p, a

    def score(self, Q):
        p, a = self.per_frame(Q); res = {o: (float(p[:, i].mean()), float(a[:, i].mean())) for i, o in enumerate(OBJ)}
        P, D = self.cfg.get('P', 1.2), self.cfg.get('D', 12.0)
        ratios = [max(pp / P, dd / D) for pp, dd in res.values()]
        if not np.isfinite(ratios).all(): return 1e6, res, (p, a)
        pk = self.cfg.get('peak', 0.0)
        base = (max(ratios) + 0.25 * sum(ratios)) if not self.cfg.get('sumcost') else (0.5 * max(ratios) + sum(pp / P + dd / D for pp, dd in res.values()) / 2)
        c = float(base + pk * sum(np.percentile(p[:, i], 95) / (4 * P) for i in range(2)))
        return c, res, (p, a)

    def evaluate(self, table, keep=False):
        try:
            Q = self.rollout(table)
            if Q is None: return dict(cost=1e6, unstable=True)
            c, res, e = self.score(Q); out = dict(cost=c, objects=res, p95=[float(np.percentile(e[0][:, i], 95)) for i in range(2)])
            if keep: out['Q'] = Q; out['err'] = e
            return out
        except Exception as ex: return dict(cost=1e6, error=repr(ex)[:300])

    def ilc_correction(self, Q, side, t0, t1, ramp=0.3):
        si = SIDES.index(side); oa = self.oa[si]; wq = self.wq[side]; corr = np.zeros((self.T, 6))
        w = sm((self.t - t0) / ramp) * (1 - sm((self.t - (t1 - ramp)) / ramp)) if t1 < self.t[-1] else sm((self.t - t0) / ramp)
        Ra = R.from_quat(Q[:, oa + 3:oa + 7], scalar_first=True); pa = Q[:, oa:oa + 3]
        Rt = R.from_quat(self.qt[:, oa + 3:oa + 7], scalar_first=True); pt = self.qt[:, oa:oa + 3]
        Rw = R.from_euler('XYZ', Q[:, wq[3:]]); pw = Q[:, wq[:3]]
        for k in range(self.T):
            if w[k] <= 0: continue
            Dr = Rt[k] * Ra[k].inv(); pn = pt[k] + Dr.apply(pw[k] - pa[k]); Rn = Dr * Rw[k]
            eu = Rn.as_euler('XYZ'); eu = Q[k, wq[3:]] + (eu - Q[k, wq[3:]] + np.pi) % (2 * np.pi) - np.pi
            corr[k, :3] = (pn - pw[k]) * w[k]; corr[k, 3:] = (eu - Q[k, wq[3:]]) * w[k]
        return corr


W = None
def init(cfg):
    global W; W = Env(cfg)
def ev(table): return W.evaluate(table)


def knot_table(env, x, t0, t1, K, chans):
    kn = np.linspace(t0, t1, K + 2); P = np.zeros((K + 2, NCH)); P[1:-1, chans] = x.reshape(K, len(chans))
    return np.stack([np.interp(env.t, kn, P[:, c]) for c in range(NCH)], 1)


def save_actor(env, table, out, info):
    out = Path(out); (out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'
    with np.load(env.ckpt, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
    d['phase_weights'] = env.weights(table); np.savez_compressed(ck, **d); np.savez_compressed(out / 'residual_table.npz', table=table)
    from learned_feedback_policy import sha
    srcrep = Path(env.ckpt).parent / 'training_report.json'
    tr = json.loads(srcrep.read_text())
    tr.update(phase_residual=dict(method='derivative-free physical fine-tuning (ILC / CEM over full-horizon closed-loop rollouts of this actor) of a time-local residual on the wrist and finger-squeeze phase weights of both hands; feedback weights, scene unchanged',
              source_checkpoint=env.ckpt, source_checkpoint_sha256=sha(env.ckpt), trainer=str(Path(__file__).resolve()), trainer_sha256=sha(__file__), info=info), checkpoint_sha256=sha(ck))
    (out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    from evaluate_policy_component import evaluate as ev_final
    ev_final(env.ep, env.scene, str(ck), str(out / 'evaluation'), str(env.epd))
    q = np.load(out / 'evaluation/execution/trajectory.npz')['qpos']
    r = json.loads((out / 'evaluation/result.json').read_text())
    print('FINAL', json.dumps(dict(proxy=r['metrics']['official_style_reference_proxy']['objects'], physics_verified=r['execution']['physics_verified'])), flush=True)


def timeline(env, e, out=None):
    p, a = e
    for s0 in range(0, env.T, 10): print(f'   {s0/20:5.1f}s', ' '.join(f"{p[s0:s0+10, i].mean():5.2f}/{a[s0:s0+10, i].mean():5.1f}" for i in range(2)))
    if out: np.savez_compressed(out, p=p, a=a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['ilc', 'cem', 'eval', 'save', 'grid'])
    ap.add_argument('--actor', required=True); ap.add_argument('--ckpt'); ap.add_argument('--out', required=True); ap.add_argument('--table')
    ap.add_argument('--lw', default=''); ap.add_argument('--rw', default=''); ap.add_argument('--iterations', type=int, default=8)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--samples', type=int, default=64); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knots', type=int, default=8); ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.03)
    ap.add_argument('--chans', default='wrist', help='wrist|all|left|right|leftall|rightall')
    ap.add_argument('--alphas', default='0.15,0.3,0.5,0.7,1.0'); ap.add_argument('--sigmas', default='0.1,0.3'); ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--grid', help='JSON list of candidates; each = list of bumps [chan, value, t0, t1, ramp]'); ap.add_argument('--P', type=float, default=1.2); ap.add_argument('--D', type=float, default=12.0); ap.add_argument('--peak', type=float, default=0.0); ap.add_argument('--sumcost', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(actor=str(Path(a.actor).resolve()), ckpt=a.ckpt, table=a.table, P=a.P, D=a.D, peak=a.peak, sumcost=a.sumcost)
    T00 = time.monotonic(); env = Env(cfg); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep.get('objects'), p95=rep.get('p95'), rollout_s=time.monotonic() - T00)), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1)); hist = []; t0 = time.monotonic()
    if a.mode == 'eval':
        timeline(env, rep['err'], out / 'timeline.npz'); return
    if a.mode == 'save':
        save_actor(env, table, out / 'actor', dict(mode='save', table=a.table, search_eval=dict(cost=rep['cost'], objects=rep.get('objects')))); return
    wins = {}
    for s, w in (('left', a.lw), ('right', a.rw)):
        if w: wins[s] = tuple(float(x) for x in w.split(','))
    best_table = table.copy(); best = dict(cost=rep['cost'], objects=rep['objects'], p95=rep['p95'])
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        if a.mode == 'grid':
            G = json.loads(Path(a.grid).read_text() if Path(a.grid).exists() else a.grid); cands = []
            for bumps in G:
                tb = table.copy()
                for ch, v, b0, b1, rp in bumps: tb[:, int(ch)] += v * sm((env.t - b0) / rp) * (1 - sm((env.t - (b1 - rp)) / rp))
                cands.append(tb)
            reps = list(pool.map(ev, cands, chunksize=1))
            for g, r in sorted(zip(G, reps), key=lambda x: x[1]['cost']): print('G', json.dumps(dict(bumps=g, cost=r['cost'], objects=r.get('objects'), p95=r.get('p95'))), flush=True)
            i = int(np.argmin([r['cost'] for r in reps]))
            if reps[i]['cost'] < best['cost']: best = dict(cost=reps[i]['cost'], objects=reps[i]['objects'], p95=reps[i]['p95'], bumps=G[i]); best_table = cands[i]
            (out / 'progress.json').write_text(json.dumps(dict(best=best, all=[dict(bumps=g, cost=r['cost'], objects=r.get('objects')) for g, r in zip(G, reps)])))
        elif a.mode == 'ilc':
            Q = rep['Q']; cur = table.copy(); curcost = rep['cost']
            alphas = [float(x) for x in a.alphas.split(',')]; sigmas = [float(x) for x in a.sigmas.split(',')]
            for it in range(a.iterations):
                corr = {s: env.ilc_correction(Q, s, *w) for s, w in wins.items()}; cands = []; meta = []
                for s_ in sigmas:
                    cs = {s: gaussian_filter1d(c, s_ * env.fps, axis=0, mode='nearest') if s_ > 0 else c for s, c in corr.items()}
                    for al in alphas:
                        for mode_ in ('full', 'pos', 'rot'):
                            for which in (list(wins) + (['both'] if len(wins) > 1 else [])):
                                dlt = np.zeros((env.T, NCH)); m_ = {'full': slice(0, 6), 'pos': slice(0, 3), 'rot': slice(3, 6)}[mode_]
                                for s in (wins if which == 'both' else [which]):
                                    si = SIDES.index(s); blk = np.zeros((env.T, 6)); blk[:, m_] = cs[s][:, m_]; dlt[:, 6 * si:6 * si + 6] = al * blk
                                cands.append(cur + dlt); meta.append(dict(alpha=al, sigma=s_, mode=mode_, side=which))
                reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); i = int(order[0])
                row = dict(iteration=it, best_candidate=meta[i], cost=reps[i]['cost'], objects=reps[i].get('objects'), p95=reps[i].get('p95'), prev_cost=curcost,
                           top=[dict(meta[j], cost=reps[j]['cost'], objects=reps[j].get('objects')) for j in order[:5]], elapsed_s=time.monotonic() - t0)
                hist.append(row); print('IT', json.dumps(row), flush=True)
                if reps[i]['cost'] < curcost:
                    cur = cands[i]; curcost = reps[i]['cost']; Q = env.evaluate(cur, keep=True)['Q']
                    best = dict(cost=curcost, objects=reps[i]['objects'], p95=reps[i]['p95']); best_table = cur.copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                else:
                    alphas = [x * 0.5 for x in alphas]
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best)))
        else:
            chans = dict(wrist=list(range(12)), all=list(range(16)), left=list(range(6)), right=list(range(6, 12)),
                         leftall=list(range(6)) + [12, 13], rightall=list(range(6, 12)) + [14, 15])[a.chans]
            t0w, t1w = [float(x) for x in (a.lw or a.rw).split(',')]
            K = a.knots; base = np.array(([a.pos_std] * 3 + [a.rot_std] * 3) * 2 + [a.sq_std] * 4)[chans]
            std = np.tile(base, K); mean = np.zeros(K * len(chans)); bx = mean.copy(); rng = np.random.default_rng(a.seed); bound = 4 * std
            for it in range(a.iterations):
                xs = np.clip(np.vstack([bx, mean, mean + rng.normal(size=(a.samples - 2, K * len(chans))) * std]), -bound, bound)
                cands = [table + knot_table(env, x, t0w, t1w, K, chans) for x in xs]
                reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = xs[order[:max(6, a.samples // 8)]]
                if reps[order[0]]['cost'] < best['cost']:
                    bx = xs[order[0]].copy(); best = dict(cost=reps[order[0]]['cost'], objects=reps[order[0]]['objects'], p95=reps[order[0]]['p95']); best_table = cands[order[0]].copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.1 * np.tile(base, K))
                row = dict(iteration=it, best=best, elapsed_s=time.monotonic() - t0); hist.append(row); print('IT', json.dumps(row), flush=True)
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best, best_x=bx.tolist())))
    np.savez_compressed(out / 'best_table.npz', table=best_table)
    if not a.no_save and best['cost'] < rep['cost']:
        save_actor(env, best_table, out / 'actor', dict(mode=a.mode, args=vars(a), search_best=best))


if __name__ == '__main__': main()

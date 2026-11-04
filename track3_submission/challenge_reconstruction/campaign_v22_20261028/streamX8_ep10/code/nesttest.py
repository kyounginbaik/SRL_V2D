"""X8 ep10: where does the beige collision proxy settle inside the blue collision proxy (MuJoCo, same solver options as the
episode scene, default cup-cup contact)?  Blue welded upright, beige dropped coaxially from rim-to-rim h = 8 cm with several
spins (sector alignments) and small lateral / tilt perturbations.  Also an analytic coaxial bound from the band profiles.
usage: nesttest.py OUTJSON t_blue_mm[,..] d_beige_mm[,..]   (t_blue 'L' = stream L blue unchanged, d 0 = beige unchanged)"""
import sys, json, tempfile, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent)); import thinlib as T
out = Path(sys.argv[1]); TB = sys.argv[2].split(','); DB = [float(x) for x in sys.argv[3].split(',')]
PR = {n: T.profile(n) for n in ('blue_cup', 'beige_cup')}
tmp = Path(tempfile.mkdtemp(prefix='x8nest_'))

def build(n, t_blue, d_beige):
    pr = PR[n]; R_in, R_out = pr['R_in'].copy(), pr['R_out'].copy()
    if n == 'blue_cup' and t_blue is not None: R_in = R_out - t_blue
    if n == 'beige_cup' and d_beige: R_in, R_out = R_in - d_beige, R_out - d_beige
    d = tmp / f'{n}_{t_blue}_{d_beige}'; d.mkdir(exist_ok=True); ps = T.parts_from(pr, R_in, R_out)
    for k, p in enumerate(ps): p.export(d / f'p{k:03d}.obj')
    com, I = T.inertia(ps); return d, len(ps), com, I, R_in, R_out

def analytic(Rb_in, Re_out):
    """coaxial min rim-to-rim h (cm) with no overlap; worst (beige vertex vs blue chord) and best (aligned) sector cases"""
    pb, pe = PR['blue_cup'], PR['beige_cup']; zb = pb['edges'] - pb['h0']; ze = pe['edges'] - pe['h0']; push = 1 / np.cos(np.pi / 24)
    res = {}
    for case, f in (('worst', push), ('aligned', 1.0)):
        for s in np.arange(-0.5, 12, 0.0002):   # s = beige h0 above blue h0 (m); increase until clear
            y = np.linspace(0, ze[-1], 400); zz = y + s; ok = zz <= zb[-1]
            if not ok.any(): break
            rb = np.interp(zz[ok], zb, Rb_in); re = np.interp(y[ok], ze, Re_out) * f
            if np.all(re <= rb) and np.all(zz[ok] >= zb[0]): res[case] = round((s + ze[-1] - zb[-1]) * 100, 3); break
    return res

def mjcf(db, nb, dbe, ne):
    pb, pe = PR['blue_cup'], PR['beige_cup']
    qb = R.align_vectors([[0, 0, 1]], [pb['a']])[0]
    meshes = ''.join(f'<mesh name="b{k}" file="{db}/p{k:03d}.obj"/>' for k in range(nb)) + ''.join(f'<mesh name="e{k}" file="{dbe}/p{k:03d}.obj"/>' for k in range(ne))
    gb = ''.join(f'<geom type="mesh" mesh="b{k}" condim="4"/>' for k in range(nb)); ge = ''.join(f'<geom type="mesh" mesh="e{k}" condim="4" mass="0"/>' for k in range(ne))
    q = qb.as_quat(scalar_first=True)
    return f'''<mujoco><size memory="256M"/><option impratio="5" integrator="implicitfast" cone="elliptic" iterations="50" noslip_iterations="5"/>
<asset>{meshes}</asset><worldbody><body name="blue" quat="{q[0]} {q[1]} {q[2]} {q[3]}">{gb}</body>
<body name="beige"><freejoint name="bf"/><inertial pos="{ICOM[0]} {ICOM[1]} {ICOM[2]}" mass="0.1" fullinertia="{II[0,0]} {II[1,1]} {II[2,2]} {II[0,1]} {II[0,2]} {II[1,2]}"/>{ge}</body></worldbody></mujoco>'''

def drop(model, spin, lat, tilt, h_start=0.08, secs=2.0):
    pb, pe = PR['blue_cup'], PR['beige_cup']; d = mujoco.MjData(model)
    Rb = R.align_vectors([[0, 0, 1]], [pb['a']])[0]
    top_b = Rb.apply(pb['c'] + pb['a'] * (pb['h1'] - 0))  # c + a*h1 is the rim (h measured from c)
    Re = R.from_rotvec([tilt, 0, 0]) * R.from_rotvec([0, 0, spin]) * R.align_vectors([[0, 0, 1]], [pe['a']])[0]
    rim_e_body = pe['c'] + pe['a'] * pe['h1']
    pos = top_b + np.array([lat, 0, h_start]) - Re.apply(rim_e_body)
    d.qpos[:3] = pos; d.qpos[3:7] = Re.as_quat(scalar_first=True)
    nstep = int(secs / model.opt.timestep); mujoco.mj_forward(model, d)
    for _ in range(nstep): mujoco.mj_step(model, d)
    Rq = R.from_quat(d.qpos[3:7], scalar_first=True); rim_e = Rq.apply(rim_e_body) + d.qpos[:3]
    ax = Rq.apply(pe['a']); off = rim_e - top_b
    return dict(h_cm=round(float(off[2] * 100), 3), lat_cm=round(float(np.linalg.norm(off[:2]) * 100), 3),
                tilt_deg=round(float(np.degrees(np.arccos(np.clip(ax[2], -1, 1)))), 2), speed=float(np.linalg.norm(d.qvel)))

res = []
for tb in TB:
    t_blue = None if tb == 'L' else float(tb) / 1000
    db, nb, *_ , Rb_in, _ = build('blue_cup', t_blue, 0)
    for dmm in DB:
        dbe, ne, ICOM, II, _, Re_out = build('beige_cup', None, dmm / 1000)
        m = mujoco.MjModel.from_xml_string(mjcf(db, nb, dbe, ne))
        drops = [dict(spin_deg=sp, lat_mm=la, tilt0_deg=tl, **drop(m, np.radians(sp), la / 1000, np.radians(tl)))
                 for sp, la, tl in ((0, 0, 0), (3.75, 0, 0), (7.5, 0, 0), (11.25, 0, 0), (5, 1.0, 0), (2, 0, 1.0), (6, 0.5, -1.0), (2, 0, 2.0), (9, -1.0, -2.0))]
        hs = [x['h_cm'] for x in drops]
        r = dict(t_blue_mm=tb, d_beige_mm=dmm, analytic=analytic(Rb_in, Re_out), mj_h_cm_median=float(np.median(hs)), mj_h_cm_min=min(hs), mj_h_cm_max=max(hs), drops=drops)
        print(json.dumps({k: v for k, v in r.items() if k != 'drops'}), flush=True); res.append(r)
out.write_text(json.dumps(res, indent=1))

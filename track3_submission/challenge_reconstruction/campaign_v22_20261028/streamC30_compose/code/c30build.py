"""C30: transplant a coworker compose_vK bundle (based on selected20_final_v8_g10b) onto our current reconstruction
selected20_final_v14_g17 (= v8_g10b + G12 ep32 spoon roll [v10_g12] + G15 clock-consistent camera + retime [v12b_g15] + G16 ep3
spoon roll [v14_g17]). Downloaded data are read as data only (np.load allow_pickle=False, json).

Per episode / object, in the RGB camera frame (camera_from_object, C):
  1. changed frames  = frames where C_vK differs from C_v8g10b (> 1e-6 m or 1e-4 deg); whole-track replacements are read from the
     coworker's own json manifests (compose_windows_v6.json / compose_whole_v7.json / compose_whole_v8.json).
  2. v10-space track: unchanged frames = C_v10g12; window frames = C_vK @ B (B = inv(C_v8g10b) C_v10g12, the G12 body
     correction, identity except ep32 spoon); whole-track objects = C_vK on every frame (theirs fully, no G12 roll).
  3. G15 exactly as build_bundle.py: world_from_rgb := G15 camera (streamG15/cam/evaluation_<e>/camera.npz),
     world_from_object = cam @ C, then retime_npz (t + delta_e, SE(3) interp; ep20 piecewise).
  4. G16: ep3 spoon right-multiplied by the constant -13.5 deg body roll X (g16_spoon_roll.json), unless ep3 spoon is a whole-track
     replacement.
  5. Objects with no change: the v14_g17 arrays are copied bit-exactly (and the chain is asserted to reproduce them to 1e-8).
usage: c30build.py VK_DIR LABEL OUT_BUNDLE OUT_TARGET REPORT.json"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/code')
from retime import retime_npz
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
G15 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15')
B8 = Path('/data1/SRL_V2D/challenge_submissions_SRL/recon_v26_selected20_final_v8_g10b')
B10 = RR / 'selected20_final_v10_g12'; B14 = RR / 'selected20_final_v14_g17'   # label only; never read or written (see below)
B12B = RR / 'selected20_final_v12b_g15'; B13 = RR / 'selected20_final_v13_g16'   # v14_g17 == v12b + ep3 object_poses from v13 (G17_CHANGES.json; checked == target_final_v14_g17)
T14 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17')
VK, LABEL, OUT, TOUT, REP = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
for p in (OUT, TOUT, REP):
    assert not p.exists(), f'{p} exists (never overwrite)'
SOURCES = [B8, B10, B12B, B13, B14, T14, VK, G15, RR / 'selected20_final_v12_g15', RR / 'selected20_final_v10_g12']
for p in (OUT, TOUT, REP):   # C30 bug guard: outputs must be new paths outside every source bundle
    rp = p.resolve()
    for src in SOURCES:
        assert not str(rp).startswith(str(src.resolve()) + '/') and rp != src.resolve(), f'{p} is inside source {src}'
def _guard(path):
    assert str(Path(path).resolve()).startswith(str(OUT.resolve()) + '/'), f'write outside OUT: {path}'
    return path
D = json.loads((G15 / 'out/sync_delta_t.json').read_text())
def ld(p): return dict(np.load(p, allow_pickle=False))
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()
whole = set()
for jf, key in (('compose_windows_v6.json', 'whole_tracks'), ('compose_whole_v7.json', 'whole_tracks'), ('compose_whole_v8.json', 'whole_tracks')):
    if (VK / jf).exists():
        for w in json.loads((VK / jf).read_text()).get(key, []): whole.add((int(w['ep']), str(w['obj'])))
def rdeg(Ra, Rb): return np.degrees(np.linalg.norm(Rot.from_matrix(np.swapaxes(Ra, -1, -2) @ Rb).as_rotvec(), axis=-1))
def runs(fr):
    out = []; fr = list(fr)
    for f in fr:
        f = int(f)
        if out and f == out[-1][1] + 1: out[-1][1] = f
        else: out.append([f, f])
    return out
shutil.copytree(B12B, OUT)   # FIX 2026-11-01: never touch the v14 bundle; OUT is rebuilt as v14 from its sources and all writes go to OUT
for f in ('object_poses.npz', 'g16_spoon_roll.json'): shutil.copy2(B13 / 'episode_000003' / f, OUT / 'episode_000003' / f)
for f in ('G16_CHANGES.json',): shutil.copy2(B13 / f, OUT / 'G16_CHANGES_source.json')
(OUT / 'G17_CHANGES.json').write_bytes((B14 / 'G17_CHANGES.json').read_bytes())
for ep in sorted(OUT.glob('episode_*')):   # OUT must equal target_final_v14_g17 exactly before any C30 change
    z = np.load(ep / 'object_poses.npz', allow_pickle=False)
    for s_, n_ in enumerate(z['names']): assert np.array_equal(np.load(T14 / ep.name / f'{n_}_trajectory.npz')['T_world_obj'], z['world_from_object'][:, s_]), (ep, n_)
rep = dict(label=LABEL, source=str(VK), base_of_source=str(B8), transplanted_onto=str(B14), whole_tracks=sorted(map(list, whole)),
           ground_truth_used=False, method=__doc__.split('\n\n')[1] if '\n\n' in __doc__ else '', episodes={})
for ep in sorted(OUT.glob('episode_*')):   # ep is inside OUT (the copy); reads and writes stay in OUT
    e = int(ep.name[-6:]); fv = VK / ep.name
    z8, z10, z14, zv = ld(B8 / ep.name / 'object_poses.npz'), ld(B10 / ep.name / 'object_poses.npz'), ld(ep / 'object_poses.npz'), ld(fv / 'object_poses.npz')
    names = [str(n) for n in z14['names']]
    assert [str(n) for n in zv['names']] == names == [str(n) for n in z8['names']], (e, zv['names'], names)
    F = len(z14['world_from_object']); assert len(zv['world_from_object']) == F == len(z8['world_from_object'])
    er = dict(world_from_rgb_equal_v8=bool(np.array_equal(zv['world_from_rgb'], z8['world_from_rgb'])),
              meshes_equal_v8={n: md5(fv / 'meshes' / f'{n}.obj') == md5(B8 / ep.name / 'meshes' / f'{n}.obj') for n in names},
              object_observed_equal_v8=bool(np.array_equal(zv['object_observed'], z8['object_observed'])),
              camera_valid_equal_v8=bool(np.array_equal(zv['camera_valid'], z8['camera_valid'])), objects={})
    # coworker C (use their camera_from_object; check world consistency)
    Cv = zv['camera_from_object'].astype(np.float64); C8 = z8['camera_from_object'].astype(np.float64); C10 = z10['camera_from_object'].astype(np.float64)
    wc = np.abs(np.einsum('tij,tnjk->tnik', zv['world_from_rgb'], Cv) - zv['world_from_object']).max(); er['vK_world_cam_consistency_max'] = float(wc)
    Cn = C10.copy(); changed_any = False; chg = {}
    for s, n in enumerate(names):
        dp = np.linalg.norm(Cv[:, s, :3, 3] - C8[:, s, :3, 3], axis=-1) * 100; dr = rdeg(C8[:, s, :3, :3], Cv[:, s, :3, :3])
        fr = np.where((dp > 1e-4) | (dr > 1e-4))[0]; isw = (e, n) in whole
        o = dict(n_changed=int(len(fr)), windows=runs(fr), whole_track=isw,
                 vs_v8_cm_max=float(dp.max()), vs_v8_deg_max=float(dr.max()), vs_v8_cm_mean_changed=float(dp[fr].mean()) if len(fr) else 0.0,
                 vs_v8_deg_mean_changed=float(dr[fr].mean()) if len(fr) else 0.0)
        if len(fr) or isw:
            changed_any = True; chg[s] = fr
            if isw: Cn[:, s] = Cv[:, s]
            else:
                Bc = np.linalg.inv(C8[fr, s]) @ C10[fr, s]; Cn[fr, s] = Cv[fr, s] @ Bc
                o['g12_body_correction_applied_deg_max'] = float(rdeg(np.eye(3)[None], Bc[:, :3, :3]).max())
        er['objects'][n] = o
    if changed_any:
        cam = np.load(G15 / f'cam/evaluation_{e}/camera.npz', allow_pickle=False)['world_from_camera'][:F]
        zz = dict(z10); zz['camera_from_object'] = Cn.astype(z10['camera_from_object'].dtype); zz['world_from_rgb'] = cam
        zz['world_from_object'] = np.einsum('tij,tnjk->tnik', cam, Cn).astype(z10['world_from_object'].dtype)
        tmpd = OUT / ep.name; src = tmpd / 'c30_v10space.npz'; dst = tmpd / 'c30_retimed.npz'
        np.savez_compressed(_guard(src), **zz); retime_npz(src, dst, D[f'evaluation_{e}']); zr = ld(dst); src.unlink(); dst.unlink()
        g16 = ep / 'g16_spoon_roll.json'
        if g16.exists() and 'wooden_spoon' in names and (e, 'wooden_spoon') not in whole:
            gi = json.loads(g16.read_text()); s = names.index('wooden_spoon'); ax = np.array(gi['axis_body']); c = np.array(gi['centroid_body'])
            Rm = Rot.from_rotvec(np.radians(gi['roll_deg']) * ax).as_matrix(); X = np.eye(4); X[:3, :3] = Rm; X[:3, 3] = c - Rm @ c
            for k in ('world_from_object', 'camera_from_object'): zr[k][:, s] = zr[k][:, s] @ X
            er['g16_roll_applied'] = gi['roll_deg']
        out = dict(z14)
        assert np.abs(zr['world_from_rgb'] - z14['world_from_rgb']).max() < 1e-8
        for s, n in enumerate(names):
            d = max(np.abs(zr[k][:, s] - z14[k][:, s]).max() for k in ('world_from_object', 'camera_from_object'))
            if s not in chg:
                assert d < 1e-8, (e, n, d)          # chain reproduces v14 exactly for untouched objects
                continue
            for k in ('world_from_object', 'camera_from_object'):
                a = out[k].copy(); a[:, s] = zr[k][:, s]; out[k] = a
            W14, Wn = z14['world_from_object'][:, s], out['world_from_object'][:, s]
            dp = np.linalg.norm(Wn[:, :3, 3] - W14[:, :3, 3], axis=-1) * 100; dr = rdeg(W14[:, :3, :3], Wn[:, :3, :3])
            fr = np.where((dp > 1e-4) | (dr > 1e-4))[0]
            er['objects'][n].update(vs_v14_frames=runs(fr), vs_v14_n=int(len(fr)), vs_v14_cm_max=float(dp.max()), vs_v14_deg_max=float(dr.max()),
                                    vs_v14_cm_mean=float(dp.mean()), vs_v14_deg_mean=float(dr.mean()),
                                    vs_v14_n_gt1cm_or_5deg=int(((dp > 1) | (dr > 5)).sum()))
        np.savez_compressed(_guard(ep / 'object_poses.npz'), **out)
        er['object_poses_md5'] = md5(ep / 'object_poses.npz')
    er['changed'] = changed_any
    rep['episodes'][e] = er
    print(e, {n: (o['n_changed'], o['whole_track'], round(o.get('vs_v14_cm_max', 0), 2), round(o.get('vs_v14_deg_max', 0), 1)) for n, o in er['objects'].items()}, flush=True)
(OUT / 'C30_CHANGES.json').write_text(json.dumps(rep, indent=1))
REP.write_text(json.dumps(rep, indent=1))
# target in the compose_target layout (same as code/build_recon_target.py)
for ep in sorted(OUT.glob('episode_*')):
    d = TOUT / ep.name; d.mkdir(parents=True)
    poses = np.load(ep / 'object_poses.npz', allow_pickle=False); meshes = json.loads((ep / 'meshes.json').read_text())
    for slot, name in enumerate(str(n) for n in poses['names']):
        np.savez_compressed(d / f'{name}_trajectory.npz', T_world_obj=poses['world_from_object'][:, slot])
        shutil.copy2(ep / meshes[name]['file'], d / f'{name}_mesh_metric.obj')
print('built', OUT, TOUT)

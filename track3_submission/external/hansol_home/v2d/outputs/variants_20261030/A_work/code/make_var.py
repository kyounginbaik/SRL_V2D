"""Make one smoother variant: public/held-out exports A_<tag> (from r8_base, in our mirror) + eval tracks (from v8_g10b,
all episodes except 32) under A_work/eval/<tag>/. Same rule for every clip. Never overwrites.
usage: make_var.py TAG 'JSON_PARAMS'"""
import sys, json, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import slib

RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
W = Path('/home/hansol/v2d/outputs/variants_20261030/A_work')
BASE = RR / 'selected20_final_v8_g10b'
SKIP_EVAL = {32}
tag, params = sys.argv[1], json.loads(sys.argv[2])


def smooth_npz(src, dst, meshes):
    d = dict(np.load(src))
    names = [str(n) for n in d['names']]
    obs = d.get('object_observed')
    wfo = d['world_from_object'].copy()
    info_out = {}
    for k, n in enumerate(names):
        info = slib.mesh_info(n, meshes[n])
        T, extra = slib.smooth_track(wfo[:, k], info, observed=None if obs is None else obs[:, k], **params)
        b = slib.summarize(slib.motion_stats(wfo[:, k], info)); a = slib.summarize(slib.motion_stats(T, info))
        dr = slib.rel_angle_deg(T[:, :3, :3], wfo[:, k, :3, :3], info)
        dp = np.linalg.norm(slib.centers(T, info) - slib.centers(wfo[:, k], info), axis=-1) * 100
        info_out[n] = dict(extra, change_rot_mean=float(dr.mean()), change_rot_max=float(dr.max()), change_pos_mean=float(dp.mean()),
                           change_pos_max=float(dp.max()), before=b, after=a)
        wfo[:, k] = T
    if 'camera_from_object' in d:
        changed = np.any(wfo != d['world_from_object'], axis=(2, 3))
        cfo = np.einsum('nij,nkjl->nkil', np.linalg.inv(d['world_from_rgb']), wfo)
        d['camera_from_object'] = np.where(changed[:, :, None, None], cfo, d['camera_from_object'])
    d['world_from_object'] = wfo
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists(), dst
    np.savez(dst, **d)
    return info_out


report = dict(tag=tag, params=params, public={}, eval={})
for split in ['public', 'public_heldout']:
    for ep in sorted((W / 'pub' / split).glob('episode_*')):
        b = ep / 'exports' / 'r8_base'; o = ep / 'exports' / f'A_{tag}'
        if (o / 'object_poses.npz').exists():
            continue
        meshes = json.loads((b / 'meshes.json').read_text())
        report['public'][f'{split}/{ep.name}'] = smooth_npz(b / 'object_poses.npz', o / 'object_poses.npz', meshes)
        for f in ['meshes.json', 'selection.json']:
            if (b / f).exists(): shutil.copy2(b / f, o / f)
for ep in sorted(BASE.glob('episode_*')):
    if int(ep.name[-6:]) in SKIP_EVAL:
        continue
    mj = json.loads((ep / 'meshes.json').read_text())
    report['eval'][ep.name] = smooth_npz(ep / 'object_poses.npz', W / 'eval' / tag / ep.name / 'object_poses.npz', {n: str(ep / v['file']) for n, v in mj.items()})
(W / 'eval' / tag).mkdir(parents=True, exist_ok=True)
(W / 'eval' / tag / 'smooth_report.json').write_text(json.dumps(report, indent=1))
tot = lambda part, key, ba: sum(o[ba][key] for c in report[part].values() for o in c.values())
for part in ['public', 'eval']:
    ch = [o['change_rot_mean'] for c in report[part].values() for o in c.values()]; cp = [o['change_pos_mean'] for c in report[part].values() for o in c.values()]
    cuts = sum(len(o['cuts']) for c in report[part].values() for o in c.values()); low = sum(len(o['low_weight_frames']) for c in report[part].values() for o in c.values())
    print(f"{tag} {part}: rot>30 {tot(part,'rot_gt30','before')}->{tot(part,'rot_gt30','after')} rot>15 {tot(part,'rot_gt15','before')}->{tot(part,'rot_gt15','after')} "
          f"pos>3 {tot(part,'pos_gt3','before')}->{tot(part,'pos_gt3','after')} pos>2 {tot(part,'pos_gt2','before')}->{tot(part,'pos_gt2','after')} "
          f"racc>20 {tot(part,'racc_gt20','before')}->{tot(part,'racc_gt20','after')} pacc>1 {tot(part,'pacc_gt1','before')}->{tot(part,'pacc_gt1','after')} | "
          f"mean change {np.mean(ch):.2f} deg / {np.mean(cp):.3f} cm (max obj {np.max(ch):.2f} / {np.max(cp):.3f}) cuts {cuts} lowW {low}")

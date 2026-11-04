"""Build a variant-A bundle: byte-identical copy of selected20_final_v8_g10b, then object_poses.npz replaced in the
episodes the guarded smoother changed (never ep32). Writes manifest.json (+ README by the caller) and never overwrites.
usage: build_bundle.py TAG OUT_NAME LABEL"""
import sys, json, shutil, hashlib
from pathlib import Path
import numpy as np
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
BASE = RR / 'selected20_final_v8_g10b'
V = Path('/home/hansol/v2d/outputs/variants_20261030')
W = V / 'A_work'
tag, name, label = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = V / name
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

shutil.copytree(BASE, OUT, copy_function=shutil.copy2)
changed = {}
for ep in sorted(OUT.glob('episode_*')):
    e = int(ep.name[-6:])
    if e == 32:
        continue
    new = W / 'eval' / tag / ep.name / 'object_poses.npz'
    a, b = np.load(BASE / ep.name / 'object_poses.npz'), np.load(new)
    assert a.files == b.files, (ep, a.files, b.files)
    for k in a.files:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, (ep, k)
        if k not in ('world_from_object', 'camera_from_object'):
            assert np.array_equal(a[k], b[k]), (ep, k)
    assert np.allclose(np.einsum('nij,nkjl->nkil', b['world_from_rgb'], b['camera_from_object']), b['world_from_object'], atol=1e-9)
    assert np.isfinite(b['world_from_object']).all()
    assert np.array_equal(a['world_from_object'][0], b['world_from_object'][0]), 'frame 0 must be unchanged'
    if np.array_equal(a['world_from_object'], b['world_from_object']):
        continue
    shutil.copy2(new, ep / 'object_poses.npz')
    rep = json.loads((W / 'eval' / tag / ep.name / 'guard_report.json').read_text())
    names = [str(n) for n in b['names']]
    objs = {}
    for k, n in enumerate(names):
        if np.array_equal(a['world_from_object'][:, k], b['world_from_object'][:, k]):
            continue
        r = rep['objects'][n]
        objs[n] = dict(change_rot_mean_deg=r['change_rot_mean'], change_rot_max_deg=r['change_rot_max'], change_pos_mean_cm=r['change_pos_mean'],
                       change_pos_max_cm=r['change_pos_max'], guard_rounds=r['rounds'], guard_anchored_frames=r['guarded_frames'],
                       kept_jumps_at_frames=r['cuts'], downweighted_spike_frames=r['low_weight_frames'],
                       steps_before={k2: r['before'][k2] for k2 in ['rot_gt30', 'rot_gt15', 'pos_gt3', 'pos_gt2', 'rot_max', 'pos_max', 'racc_gt20']},
                       steps_after={k2: r['after'][k2] for k2 in ['rot_gt30', 'rot_gt15', 'pos_gt3', 'pos_gt2', 'rot_max', 'pos_max', 'racc_gt20']})
    changed[ep.name] = dict(objects=objs, sha256_before=sha(BASE / ep.name / 'object_poses.npz'), sha256_after=sha(ep / 'object_poses.npz'))

files = {}
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        rel = str(p.relative_to(OUT)); bp = BASE / rel
        files[rel] = dict(sha256=sha(p), identical_to_v8_g10b=bp.exists() and sha(bp) == sha(p))
params = json.loads((W / 'eval' / tag / 'episode_000001' / 'guard_report.json').read_text())['params']
pub = json.loads((W / f'compare_{tag}.json').read_text()) if (W / f'compare_{tag}.json').exists() else None
qc = json.loads((W / f'qc_eval_{tag}.json').read_text())['summary'] if (W / f'qc_eval_{tag}.json').exists() else None
qcp = json.loads((W / f'qc_eval_{tag}.json').read_text())['objects'] if (W / f'qc_eval_{tag}.json').exists() else None
man = dict(variant=f'A_{label}', base='selected20_final_v8_g10b', base_path=str(BASE), ground_truth_used=False,
           eval_gt_read=False, meshes='unchanged (byte-identical to v8_g10b)', episode_32='byte-identical to v8_g10b (owned by streams G12/H12)',
           rule='image-guarded robust SE(3) smoother (A_work/code/slib.py smooth_track + guard_clip.py), one rule for every clip and object; frame 0 unchanged',
           params=params, internal_tag=tag, changed_episodes=changed,
           unchanged_episodes=sorted(ep.name for ep in OUT.glob('episode_*') if ep.name not in changed),
           public_validation=pub, eval_qc_summary=qc, eval_qc_per_object=qcp, files=files,
           code='/home/hansol/v2d/outputs/variants_20261030/A_work/code', submitted=False)
(OUT / 'manifest.json').write_text(json.dumps(man, indent=1) + '\n')
print('built', OUT, 'changed', len(changed), 'episodes;', sum(len(v['objects']) for v in changed.values()), 'objects')

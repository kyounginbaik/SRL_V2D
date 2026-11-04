"""Build a variant-A14 bundle: byte-identical copy of selected20_final_v14_g17, then object_poses.npz replaced in the
episodes the guarded smoother changed (never ep32). Writes manifest.json; never overwrites.
usage: build_bundle14.py TAG OUT_NAME LABEL"""
import sys, json, shutil, hashlib
from pathlib import Path
import numpy as np
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
BASE = RR / 'selected20_final_v14_g17'
V = Path('/home/hansol/v2d/outputs/variants_20261030')
W = V / 'A_work'
tag, name, label = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = V / name
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
KEYS = ['rot_gt30', 'rot_gt15', 'pos_gt3', 'pos_gt2', 'rot_max', 'pos_max', 'racc_gt20']

shutil.copytree(BASE, OUT, copy_function=shutil.copy2)
changed = {}
for ep in sorted(OUT.glob('episode_*')):
    if int(ep.name[-6:]) == 32:
        continue
    new = W / 'eval14' / tag / ep.name / 'object_poses.npz'
    a, b = np.load(BASE / ep.name / 'object_poses.npz'), np.load(new)
    assert a.files == b.files, (ep, a.files, b.files)
    for k in a.files:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, (ep, k)
        if k not in ('world_from_object', 'camera_from_object'):
            assert np.array_equal(a[k], b[k]), (ep, k)
    assert np.isfinite(b['world_from_object']).all() and np.isfinite(b['camera_from_object']).all()
    assert np.array_equal(a['world_from_object'][0], b['world_from_object'][0]) and np.array_equal(a['camera_from_object'][0], b['camera_from_object'][0]), 'frame 0 must be unchanged'
    # the body-frame change applied to cfo equals the one applied to wfo
    dw = np.linalg.inv(a['world_from_object']) @ b['world_from_object']; dc = np.linalg.inv(a['camera_from_object']) @ b['camera_from_object']
    assert np.abs(dw - dc).max() < 1e-9, (ep, np.abs(dw - dc).max())
    if np.array_equal(a['world_from_object'], b['world_from_object']):
        continue
    shutil.copy2(new, ep / 'object_poses.npz')
    rep = json.loads((W / 'eval14' / tag / ep.name / 'guard_report.json').read_text())
    objs = {}
    for k, n in enumerate(str(x) for x in b['names']):
        if np.array_equal(a['world_from_object'][:, k], b['world_from_object'][:, k]):
            continue
        r = rep['objects'][n]
        objs[n] = dict(change_rot_mean_deg=r['change_rot_mean'], change_rot_max_deg=r['change_rot_max'], change_pos_mean_cm=r['change_pos_mean'],
                       change_pos_max_cm=r['change_pos_max'], guard_rounds=r['rounds'], guard_anchored_frames=r['guarded_frames'], guard_final_failing=r['final_bad'],
                       kept_jumps_at_frames=r['cuts'], downweighted_spike_frames=r['low_weight_frames'],
                       steps_before={k2: r['before'][k2] for k2 in KEYS}, steps_after={k2: r['after'][k2] for k2 in KEYS})
    changed[ep.name] = dict(objects=objs, delta_frames=rep['delta_values'], sha256_before=sha(BASE / ep.name / 'object_poses.npz'), sha256_after=sha(ep / 'object_poses.npz'))

files = {}
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        rel = str(p.relative_to(OUT)); bp = BASE / rel
        files[rel] = dict(sha256=sha(p), identical_to_v14_g17=bp.exists() and sha(bp) == sha(p))
params = json.loads((W / 'eval14' / tag / 'episode_000001' / 'guard_report.json').read_text())['params']
J = lambda f: json.loads((W / f).read_text()) if (W / f).exists() else None
cq, nq = J(f'clockqc14_{tag}.json'), J(f'qc_eval14_{tag}.json')
man = dict(variant=f'A14_{label}', base='selected20_final_v14_g17', base_path=str(BASE), ground_truth_used=False, eval_gt_read=False,
           meshes='unchanged (byte-identical to v14_g17)', episode_32='byte-identical to v14_g17 (owned by streams G12/H12)',
           rule='same variant-A rule as A_smooth_* (slib.smooth_track + image guard), with clock-consistent evidence for the G15 re-timed poses '
                '(guard_clip14.py: IoU and ball on the RGB clock = track shifted by -delta, depth residual on the stereo clock = track as stored); frame 0 unchanged',
           params=params, internal_tag=f'14_{tag}', changed_episodes=changed,
           unchanged_episodes=sorted(ep.name for ep in OUT.glob('episode_*') if ep.name not in changed),
           public_validation=dict(vs_g15_camrt=J(f'compare14_A14_{tag}_vs_g15_camrt.json'), vs_r8_base=J(f'compare14_A14_{tag}_vs_r8_base.json')),
           eval_qc_clock_consistent=cq, eval_qc_unchanged_qcpy=dict(summary=nq['summary'], objects=nq['objects'], note='qc.py compares stored (stereo-clock) poses with RGB-clock masks') if nq else None,
           files=files, code='/home/hansol/v2d/outputs/variants_20261030/A_work/code', submitted=False)
(OUT / 'manifest.json').write_text(json.dumps(man, indent=1) + '\n')
print('built', OUT, 'changed', len(changed), 'episodes;', sum(len(v['objects']) for v in changed.values()), 'objects')

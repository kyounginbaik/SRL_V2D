"""Build recon_E14_best_B2A: B14_family_consensus (= v14_g17 + B2 poses in 14 eps) with the A14-mild guarded smoother applied on
top of the B2 tracks (A_work/eval14/B2A_g8) in all eps except 32, and episode_000032 replaced entirely by
recon_E14_ep32_annot_roll/episode_000032 (L3b). B14's own top-level README/manifest are kept as README_base_B2.md / manifest_base_B2.json.
Writes manifest.json (sha256 of every file vs v14_g17 and vs B14_family_consensus); never overwrites.
usage: build_b2a.py"""
import json, shutil, hashlib
from pathlib import Path
import numpy as np
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
V14 = RR / 'selected20_final_v14_g17'
V = Path('/home/hansol/v2d/outputs/variants_20261030')
B2 = V / 'B14_family_consensus'
EP32 = V / 'recon_E14_ep32_annot_roll' / 'episode_000032'
W = V / 'A_work'
tag, name = 'B2A_g8', 'recon_E14_best_B2A'
OUT = V / name
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
KEYS = ['rot_gt30', 'rot_gt15', 'pos_gt3', 'pos_gt2', 'rot_max', 'pos_max', 'racc_gt20']

shutil.copytree(B2, OUT, copy_function=shutil.copy2, ignore=lambda d, fs: ['README.md', 'manifest.json'] if Path(d) == B2 else [])
shutil.copy2(B2 / 'README.md', OUT / 'README_base_B2.md'); shutil.copy2(B2 / 'manifest.json', OUT / 'manifest_base_B2.json')
shutil.rmtree(OUT / 'episode_000032'); shutil.copytree(EP32, OUT / 'episode_000032', copy_function=shutil.copy2)
changed = {}
for ep in sorted(OUT.glob('episode_*')):
    if int(ep.name[-6:]) == 32:
        continue
    new = W / 'eval14' / tag / ep.name / 'object_poses.npz'
    a, b, v = np.load(B2 / ep.name / 'object_poses.npz'), np.load(new), np.load(V14 / ep.name / 'object_poses.npz')
    assert a.files == b.files == v.files, (ep, a.files, b.files)
    for k in a.files:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, (ep, k)
        if k not in ('world_from_object', 'camera_from_object'):
            assert np.array_equal(a[k], b[k]) and np.array_equal(v[k], b[k]), (ep, k)
    assert np.isfinite(b['world_from_object']).all() and np.isfinite(b['camera_from_object']).all()
    assert np.array_equal(a['world_from_object'][0], b['world_from_object'][0]) and np.array_equal(a['camera_from_object'][0], b['camera_from_object'][0]), 'frame 0 must equal the B2 base'
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
                       steps_before_B2={k2: r['before'][k2] for k2 in KEYS}, steps_after={k2: r['after'][k2] for k2 in KEYS})
    changed[ep.name] = dict(objects=objs, delta_frames=rep['delta_values'], B2_changed_this_episode=sha(B2 / ep.name / 'object_poses.npz') != sha(V14 / ep.name / 'object_poses.npz'),
                            frame0_equals_v14=bool(np.array_equal(v['world_from_object'][0], b['world_from_object'][0])),
                            sha256_v14=sha(V14 / ep.name / 'object_poses.npz'), sha256_B2=sha(B2 / ep.name / 'object_poses.npz'), sha256_after=sha(ep / 'object_poses.npz'))

files = {}
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        rel = str(p.relative_to(OUT)); s = sha(p)
        files[rel] = dict(sha256=s, identical_to_v14_g17=(V14 / rel).exists() and sha(V14 / rel) == s, identical_to_B14_consensus=(B2 / rel).exists() and sha(B2 / rel) == s)
J = lambda f: json.loads((W / f).read_text()) if (W / f).exists() else None
params = json.loads((W / 'eval14' / tag / 'episode_000001' / 'guard_report.json').read_text())['params']
man = dict(variant='recon_E14_best_B2A', components=dict(B2='B14_family_consensus (variant B, soft per-frame family consensus, 14 eps)',
                                                        A14_mild='variant A guarded Whittaker smoother, g8 settings, applied on top of the B2 tracks',
                                                        ep32='recon_E14_ep32_annot_roll/episode_000032 (L3b spoon roll from the manual annotation), copied whole'),
           base='selected20_final_v14_g17', base_path=str(V14), pose_base=str(B2), ground_truth_used=False, eval_gt_read=False,
           meshes='unchanged (byte-identical to v14_g17)',
           episode_32=dict(source=str(EP32), sha256_object_poses=sha(OUT / 'episode_000032' / 'object_poses.npz'), smoothed=False),
           rule='guard_clip14.py (slib.smooth_track lam 8 + image guard: IoU/ball on the RGB clock = track shifted by -delta, depth on the stereo clock = as stored), '
                'input = B2 poses; frame 0 equal to the B2 base', params=params, changed_episodes=changed,
           unchanged_episodes_vs_B2=sorted(ep.name for ep in OUT.glob('episode_*') if ep.name not in changed),
           changed_files_vs_v14_g17=sorted(k for k, x in files.items() if not x['identical_to_v14_g17']),
           public_validation=dict(vs_g15_camrt=J('compare14_A14_B2A_g8_vs_g15_camrt.json'), vs_v14_B2=J('compare14_A14_B2A_g8_vs_v14_B2.json'),
                                  vs_A14_mild=J('compare14_A14_B2A_g8_vs_A14_g8.json')),
           eval_qc_clock_consistent=J('clockqc_b2a_B2A_g8.json'), files=files, code=str(W / 'code'), submitted=False)
(OUT / 'manifest.json').write_text(json.dumps(man, indent=1) + '\n')
print('built', OUT, 'changed vs B2:', len(changed), 'episodes;', sum(len(x['objects']) for x in changed.values()), 'objects; files', len(files),
      'identical to v14', sum(x['identical_to_v14_g17'] for x in files.values()), 'changed vs v14:', man['changed_files_vs_v14_g17'])

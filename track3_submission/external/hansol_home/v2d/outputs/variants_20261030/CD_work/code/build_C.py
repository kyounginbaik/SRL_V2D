"""Build variant C (ep20 only): v8_g10b with episode 20's camera.npz + object_poses.npz taken from selected20_final_v12b_g15
(G15 GT-free clock fix: poses resampled onto the stereo/GT clock, piecewise delta 0.925 frame for f0-52 and 1.85 from f53,
plus the clock-consistent camera path). Every other file is copied byte-identically from v8_g10b. Writes manifest.json.
usage: build_C.py OUT_DIR"""
import sys, os, json, shutil, hashlib, numpy as np
from scipy.spatial.transform import Rotation as R

RR = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'
BASE, SRC = f'{RR}/selected20_final_v8_g10b', f'{RR}/selected20_final_v12b_g15'
G15 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15'
OUT = sys.argv[1]
EP = 'episode_000020'
CHANGED = [f'{EP}/camera.npz', f'{EP}/object_poses.npz']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

assert not os.path.exists(OUT), f'{OUT} exists; refusing to overwrite'
shutil.copytree(BASE, OUT, copy_function=shutil.copy2)
for rel in CHANGED:
    shutil.copy2(f'{SRC}/{rel}', f'{OUT}/{rel}')

# file-level audit: every base file is present; only CHANGED differ; no extra files
base_files = sorted(os.path.relpath(os.path.join(d, f), BASE) for d, _, fs in os.walk(BASE) for f in fs)
out_files = sorted(os.path.relpath(os.path.join(d, f), OUT) for d, _, fs in os.walk(OUT) for f in fs)
assert base_files == out_files, 'file set differs from v8_g10b'
diff = [r for r in base_files if sha(f'{BASE}/{r}') != sha(f'{OUT}/{r}')]
assert sorted(diff) == sorted(CHANGED), diff
meshes = {r: sha(f'{OUT}/{r}') for r in base_files if r.endswith('.obj')}
assert all(sha(f'{BASE}/{r}') == h for r, h in meshes.items())
# v12b's other ep20 files equal v8_g10b's (so the two swapped files are the whole G15 change for ep20, apart from its extra notes)
src_same = {r: sha(f'{SRC}/{r}') == sha(f'{BASE}/{r}') for r in base_files if r.startswith(EP + '/') and r not in CHANGED}
assert all(src_same.values()), src_same

# pose-level description of the change (GT-free)
a, b = np.load(f'{BASE}/{EP}/object_poses.npz'), np.load(f'{OUT}/{EP}/object_poses.npz')
assert a.files == b.files and all(a[k].shape == b[k].shape for k in a.files)
names = [str(n) for n in a['names']]
Wb = b['world_from_object']; cons = np.einsum('fij,fojk->foik', b['world_from_rgb'], b['camera_from_object'])
consistency = float(np.abs(cons - Wb).max())
sync = json.load(open(f'{G15}/out/sync_delta_t_summary.json'))['evaluation_20']
F = len(Wb); dl = np.zeros(F)
for v, lo, hi in zip(sync['values'], sync['bounds'][:-1], sync['bounds'][1:]): dl[lo:hi] = v
t = np.clip(np.arange(F) + dl, 0, F - 1)
objs = {}
for o, nm in enumerate(names):
    C = a['camera_from_object'][:, o]
    pos = np.array([np.interp(t, np.arange(F), C[:, i, 3]) for i in range(3)]).T
    from scipy.spatial.transform import Slerp
    rot = Slerp(np.arange(F), R.from_matrix(C[:, :3, :3]))(t)
    res_p = float(np.linalg.norm(b['camera_from_object'][:, o, :3, 3] - pos, axis=1).max() * 100)
    res_r = float((R.from_matrix(b['camera_from_object'][:, o, :3, :3]) * rot.inv()).magnitude().max() * 57.2958)
    dw = np.linalg.norm(Wb[:, o, :3, 3] - a['world_from_object'][:, o, :3, 3], axis=1) * 100
    dr = (R.from_matrix(Wb[:, o, :3, :3]) * R.from_matrix(a['world_from_object'][:, o, :3, :3]).inv()).magnitude() * 57.2958
    objs[nm] = dict(frames_changed=f'0-{F - 1} (all {F})', world_shift_cm_mean=round(float(dw.mean()), 3), world_shift_cm_max=round(float(dw.max()), 3),
                    world_rot_deg_mean=round(float(dr.mean()), 3), world_rot_deg_max=round(float(dr.max()), 3),
                    camera_from_object_equals_base_resampled_at_t_plus_delta=dict(max_cm=round(res_p, 5), max_deg=round(res_r, 5)))
ca, cb = np.load(f'{BASE}/{EP}/camera.npz'), np.load(f'{OUT}/{EP}/camera.npz')
cam_dp = np.linalg.norm(cb['world_from_camera'][:, :3, 3] - ca['world_from_camera'][:, :3, 3], axis=1) * 100
cam_dr = (R.from_matrix(cb['world_from_camera'][:, :3, :3]) * R.from_matrix(ca['world_from_camera'][:, :3, :3]).inv()).magnitude() * 57.2958

man = dict(
    variant='C_ep20', created='2026-10-30', author='hansol (Claude Code session, variant C/D)',
    base_bundle=BASE, ep20_source_bundle=SRC,
    rule=('G15 GT-free clock fix applied to episode 20 only: object poses resampled onto the stereo (ego_cam_b/c) clock, which is the '
          'public GT clock, with the piecewise eval20 delta 0.925 frame for f0-52 and 1.85 frame from f53 (RGB frame drop), plus the '
          'clock-consistent camera path. Taken unchanged from selected20_final_v12b_g15 (streamG15).'),
    changed_files={r: dict(sha256_base=sha(f'{BASE}/{r}'), sha256_new=sha(f'{OUT}/{r}'), source=f'{SRC}/{r}') for r in CHANGED},
    not_copied_from_source=['episode_000020/camera_pre_g15.npz', 'episode_000020/camera_report_g15.json'],
    changed_objects=objs,
    camera_path_change=dict(mean_cm=round(float(cam_dp.mean()), 3), max_cm=round(float(cam_dp.max()), 3), mean_deg=round(float(cam_dr.mean()), 3), max_deg=round(float(cam_dr.max()), 3)),
    world_equals_world_from_rgb_times_camera_from_object_max_abs=consistency,
    sync_profile_eval20=sync,
    unchanged=dict(n_files=len(base_files) - len(CHANGED), note='every other file sha256-identical to v8_g10b, including all meshes and bundle_manifest.json',
                   meshes_sha256=meshes),
    bundle_manifest_note='bundle_manifest.json is v8_g10b byte-identical (its sha256 table is the legacy final_v6_z9 table, as in v8_g10b); this manifest.json lists the changed files.',
    ground_truth_used=False, scan_meshes_used=False, meshes_changed=False,
)
json.dump(man, open(f'{OUT}/manifest.json', 'w'), indent=1)
print(json.dumps({k: man[k] for k in ('changed_files', 'changed_objects', 'camera_path_change', 'world_equals_world_from_rgb_times_camera_from_object_max_abs')}, indent=1))
print('files', len(out_files), 'changed', diff, 'meshes identical', len(meshes))

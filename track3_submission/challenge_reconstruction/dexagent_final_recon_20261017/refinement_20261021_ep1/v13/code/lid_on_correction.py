"""Episode 1 diagnostic / corrected reference: while the lid sits on the pot (reference lid high: t<SEG1_END and t>=SEG2_START),
the reconstructed pot is tilted ~7-15 deg (stereo ICP of the real depth wants a 10-18 deg correction at frames 0-50 and our
flat-seated physics pot fits the depth better). Replace the pot orientation in those segments by its orientation at the nearest
lid-off on-table frame and carry the lid rigidly about the pot anchor; smooth 0.25 s blends. Positions of the pot anchor unchanged.
usage: lid_on_correction.py SRC_DIR OUT_DIR [seg1_end seg1_ref seg2_start seg2_ref]"""
import sys, json, shutil, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
src, out = Path(sys.argv[1]), Path(sys.argv[2]); a = list(map(float, sys.argv[3:7])) if len(sys.argv) > 3 else [3.4, 4.0, 12.9, 12.3]
s1e, s1r, s2s, s2r = a; fps = float(json.load(open(src / 'manifest.json'))['fps'])
z = dict(np.load(src / 'object_poses.npz', allow_pickle=False)); names = [str(x) for x in z['names']]; T = z['world_from_object'].copy()
ip, il = names.index('white_pot'), names.index('white_pot_lid'); n = len(T); t = np.arange(n) / fps
geo = json.load(open(sys.argv[7])) if len(sys.argv) > 7 else None
anchor = np.asarray(geo['objects']['white_pot']['anchor_local_m']) if geo else np.zeros(3)
def w(x):  # weight of the correction: 1 inside lid-on segments, smooth 0.25 s ramps
    r = .25; s = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
    return np.maximum(1 - s((x - s1e) / r), s((x - s2s) / r))
Rk1, Rk2 = T[int(round(s1r * fps)), ip, :3, :3], T[int(round(s2r * fps)), ip, :3, :3]
new = T.copy()
for k in range(n):
    wk = float(w(t[k]));
    if wk <= 0: continue
    Rold = T[k, ip, :3, :3]; target = Rk1 if t[k] < (s1e + s2s) / 2 else Rk2
    D = R.from_matrix(target @ Rold.T); D = R.from_rotvec(D.as_rotvec() * wk).as_matrix()
    c = Rold @ anchor + T[k, ip, :3, 3]                       # pot anchor (world) stays fixed
    for j in (ip, il):
        new[k, j, :3, :3] = D @ T[k, j, :3, :3]; new[k, j, :3, 3] = D @ (T[k, j, :3, 3] - c) + c
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != 'object_poses.npz' and f.is_file(): shutil.copy2(f, out / f.name)
z['world_from_object'] = new; np.savez(out / 'object_poses.npz', **z)
rel = [np.degrees(R.from_matrix(new[k, ip, :3, :3] @ T[k, ip, :3, :3].T).magnitude()) for k in range(n)]
rec = dict(source=str(src), segments=dict(seg1_end_s=s1e, seg1_orientation_from_s=s1r, seg2_start_s=s2s, seg2_orientation_from_s=s2r, ramp_s=.25),
           frames_changed=int(np.sum(np.array(rel) > 1e-6)), max_pot_rotation_change_deg=float(max(rel)), frame0_rotation_change_deg=float(rel[0]),
           evidence='refinement_20261021_ep1/stereo_icp_check.json + stereo_icp_check_nolid.json (10-18 deg ICP correction at frames 0-50; flat seated pot fits depth better)')
json.dump(rec, open(out / 'lid_on_correction.json', 'w'), indent=1); print(json.dumps(rec))

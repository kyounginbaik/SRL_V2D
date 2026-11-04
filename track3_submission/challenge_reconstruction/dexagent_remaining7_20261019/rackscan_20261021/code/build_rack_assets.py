"""Replace the plastic_dish_rack geometry of one rack episode with the scan-derived mesh, expressed in OUR rack body frame.
Placement: kit registration (MP-SR._register_object_body algorithm, same seed) between our submitted rack mesh and the scan.
Among near-equal registration starts (<= best + 0.1 cm residual) choose the one that rests the rack flat on the table at
frame 0 (scan rest-up axis measured from 5 public GT episodes). Real (scan) size is kept; the mesh is lifted along world z so
its frame-0 bottom equals the old mesh's bottom. Then CoACD (same settings as the old repair) and a URDF with inertia from
the convex parts (mass unchanged 0.1 kg). Writes a full source episode copy under OUT/collision_recon/episode_N.
usage: build_rack_assets.py EPISODE OUT_ROOT"""
import sys, json, math, shutil, hashlib, importlib.util, io, base64, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, trimesh, pyarrow.parquet as pq
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
ep = int(sys.argv[1]); OUT = Path(sys.argv[2]); name = f'episode_{ep:06d}'
R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
V6 = Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_pot_fix_v6_20261020/all_20_episodes')
SCAN = Path('/data1/SRL_V2D/gt_best_per_object/plastic_dish_rack.ply')
UP_SCAN = np.array([0.997, -0.017, -0.070]); UP_SCAN /= np.linalg.norm(UP_SCAN)   # world z in scan frame, public eps 12/13/26/27/31
spec = importlib.util.spec_from_file_location('mpsr', V6 / 'submission_kit/metric_code/track_3/MP-SR.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def align_all(reference, candidate):
    tree = cKDTree(reference); out = []
    for start in M._initial_rotations(reference, candidate, M._ALIGNMENT_STARTS):
        rotation = start.copy(); translation = reference.mean(0) - rotation @ candidate.mean(0); previous = math.inf
        for _ in range(M._ICP_ITERATIONS):
            moved = candidate @ rotation.T + translation; d, idx = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(d) * M._TRIM_FRACTION))); chosen = np.argpartition(d, keep - 1)[:keep] if keep < len(d) else np.arange(len(d))
            sr, st = M._kabsch(moved[chosen], reference[idx[chosen]]); rotation = sr @ rotation; translation = sr @ translation + st
            err = float(np.mean(d[chosen]))
            if math.isfinite(previous) and abs(previous - err) <= 1e-8 * max(previous, 1.0): break
            previous = err
        out.append((rotation, translation, M._symmetric_mean_distance(reference, candidate @ rotation.T + translation)))
    return sorted(out, key=lambda t: t[2])

# our submitted (budgeted) rack mesh exactly as the scorer decodes it from the v6 parquet
roster = json.load(open(V6 / 'submission_kit/data/track_3_evaluation_objects.json'))['episodes'][str(ep)]; slot = roster.index('plastic_dish_rack')
sub = pq.read_table(V6 / 'track3_submission.parquet', columns=['row_id', 'object_mesh_b64']).to_pandas()
key = f'track_3/e{ep:06d}/'; rows = sub[sub.row_id.str.startswith(key) & sub.row_id.str.endswith(f'/o{slot}') & (sub.object_mesh_b64.fillna('').str.len() > 0)]
cv, cf = M._decode_object_mesh(rows.object_mesh_b64.iloc[0]); cv = np.asarray(cv, float); cf = np.asarray(cf, np.int64)
scan = trimesh.load(SCAN, force='mesh', process=False); rv, rf = np.asarray(scan.vertices, float), np.asarray(scan.faces, np.int64)
cc, cr = M._surface_moments(cv, cf); rc, rr = M._surface_moments(rv, rf); s = rr / cr; seed = 1_000_003 * ep + 101 * slot
kit_anchor, kit_rot = M._register_object_body((cv, cf), (rv, rf), rc, seed)        # what the scorer did for v6
cp = (M._sample_mesh_surface(cv, cf, 3000, seed) - cc) * s; rp = M._sample_mesh_surface(rv, rf, 3000, seed) - rc
sols = align_all(rp, cp); best = sols[0]
src = R7 / 'collision_recon' / name
with np.load(src / 'object_poses.npz', allow_pickle=False) as z:
    names = [str(x) for x in z['names']]; T0 = z['world_from_object'][0, names.index('plastic_dish_rack')]
Rw0, p0 = T0[:3, :3], T0[:3, 3]
cands = []
for i, (rot, off, res) in enumerate(sols):
    if res - best[2] > 0.001: continue
    up_world = Rw0 @ rot.T @ UP_SCAN; tilt = float(np.degrees(np.arccos(np.clip(up_world[2], -1, 1))))
    rel = float(np.degrees((R.from_matrix(rot) * R.from_matrix(best[0]).inv()).magnitude())); kitrel = float(np.degrees((R.from_matrix(rot) * R.from_matrix(kit_rot).inv()).magnitude()))
    cands.append(dict(index=i, residual_cm=res * 100, tilt_deg=tilt, rot_from_best_deg=rel, rot_from_kit_v6_deg=kitrel, rot=rot, off=off))
chosen = min(cands, key=lambda c: c['tilt_deg'])
rot, off = chosen['rot'], chosen['off']; c0 = cc + rot.T @ (-off) / s
Vb = c0 + (rv - rc) @ rot        # = c0 + rot^T (X_scan - rc), real scan size
old = trimesh.load(src / 'assets/plastic_dish_rack_visual.obj', force='mesh', process=False)
old_bottom = float((np.asarray(old.vertices) @ Rw0.T + p0)[:, 2].min()); new_bottom = float((Vb @ Rw0.T + p0)[:, 2].min())
lift = Rw0.T @ np.array([0, 0, old_bottom - new_bottom]); Vb = Vb + lift
dst = OUT / 'collision_recon' / name
if dst.exists(): raise SystemExit('refusing to overwrite ' + str(dst))
shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.mp4', '__pycache__'))
A = dst / 'assets'
for d in ('plastic_dish_rack_convex', 'plastic_dish_rack_convex_repaired'): shutil.rmtree(A / d, ignore_errors=True)
mesh = trimesh.Trimesh(Vb, rf, process=False)
for f in ('plastic_dish_rack_visual.obj', 'plastic_dish_rack.obj'): mesh.export(A / f)
mesh.export(A / 'plastic_dish_rack.ply')
import coacd
settings = dict(threshold=0.01, max_convex_hull=128, preprocess_mode='auto', preprocess_resolution=100, resolution=2000, mcts_nodes=10,
                mcts_iterations=50, mcts_max_depth=3, merge=True, decimate=False, seed=44)
parts = coacd.run_coacd(coacd.Mesh(Vb, rf), **settings); P = A / 'plastic_dish_rack_convex_repaired'; P.mkdir()
hulls = []
for k, (v, f) in enumerate(parts):
    h = trimesh.Trimesh(v, f, process=True).convex_hull; h.export(P / f'part_{k:03d}.obj'); hulls.append(h)
vol = np.array([h.volume for h in hulls]); mass = 0.1; rho = mass / vol.sum()
com = sum(h.volume * h.center_mass for h in hulls) / vol.sum(); I = np.zeros((3, 3))
for h in hulls:
    m_i = rho * h.volume; Ii = h.moment_inertia / h.volume * m_i if h.volume > 0 else np.zeros((3, 3)); d = h.center_mass - com
    I += Ii + m_i * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
tree = ET.parse(src / 'assets/plastic_dish_rack.urdf'); link = tree.getroot().find('link'); inert = link.find('inertial')
inert.find('origin').set('xyz', ' '.join(repr(float(x)) for x in com)); inert.find('mass').set('value', '0.1')
for k, ax in zip(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))): inert.find('inertia').set(k, repr(float(I[ax])))
for c in list(link.findall('collision')): link.remove(c)
for k in range(len(hulls)):
    c = ET.SubElement(link, 'collision'); g = ET.SubElement(c, 'geometry'); ET.SubElement(g, 'mesh', filename=f'plastic_dish_rack_convex_repaired/part_{k:03d}.obj')
ET.indent(tree); tree.write(A / 'plastic_dish_rack.urdf', encoding='utf-8', xml_declaration=True)
rep = json.loads((A / 'asset_report.json').read_text()); rep['plastic_dish_rack'].update(mesh_sha256=sha(A / 'plastic_dish_rack_visual.obj'), faces=int(len(rf)), source=str(SCAN),
    convex_parts=len(hulls), convex_method='CoACD 1.0.7 (same settings as the earlier collision repair)', volume_cm3=float(vol.sum() * 1e6), cleaned=False)
(A / 'asset_report.json').write_text(json.dumps(rep, indent=2))
info = dict(episode=ep, slot=slot, scan=str(SCAN), scan_sha256=sha(SCAN), scale_scan_over_ours=float(s), body_frame='ours (object poses unchanged)',
    candidates=[{k: v for k, v in c.items() if k not in ('rot', 'off')} for c in cands], chosen_index=chosen['index'],
    chosen_rule='lowest frame-0 tilt of the scan rest-up axis among starts within 0.1 cm of the best residual',
    lift_world_z_cm=float((old_bottom - new_bottom) * 100), frame0_bottom_m=old_bottom, convex_parts=len(hulls), inertia_com=com.tolist(),
    new_visual_sha256=sha(A / 'plastic_dish_rack_visual.obj'), old_visual_sha256=sha(src / 'assets/plastic_dish_rack_visual.obj'))
(dst / 'rack_scan_provenance.json').write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k in ('episode', 'candidates', 'chosen_index', 'lift_world_z_cm', 'convex_parts', 'scale_scan_over_ours')}, indent=1))

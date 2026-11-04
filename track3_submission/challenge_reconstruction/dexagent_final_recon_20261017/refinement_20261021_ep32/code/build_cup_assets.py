"""Episode 32: replace blue_cup geometry by the scan-derived meshes expressed in OUR cup body frames (object poses
unchanged). Placement = kit registration (MP-SR._register_object_body algorithm, scorer seed) of the scan onto our submitted
mesh; among near-equal starts (<= best + 0.1 cm) keep the twin whose wide (rim) end points the same way as ours, then the best
residual. Real scan size; lifted along world z so the frame-0 bottom equals the old mesh bottom. CoACD (rack settings) for
collision. Writes a source copy (OUT/source/episode_000010) and an episode-dir copy (OUT/episode_dir/episode_000010)."""
import sys, json, math, shutil, hashlib, importlib.util, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, trimesh, pyarrow.parquet as pq
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
OUT = Path(sys.argv[1]); EP = 32; name_ep = 'episode_000032'
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
EPD = R0 / 'refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032'
V8 = Path('/data1/SRL_V2D/challenge_submissions/track3_v8_20261021/all_20_episodes')
spec = importlib.util.spec_from_file_location('mpsr', V8 / 'submission_kit/metric_code/track_3/MP-SR.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def align_all(reference, candidate):
    tree = cKDTree(reference); out = []
    for start in M._initial_rotations(reference, candidate, M._ALIGNMENT_STARTS):
        rotation = start.copy(); translation = reference.mean(0) - rotation @ candidate.mean(0); previous = math.inf
        for _ in range(M._ICP_ITERATIONS):
            moved = candidate @ rotation.T + translation; d, idx = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(d) * M._TRIM_FRACTION))); ch = np.argpartition(d, keep - 1)[:keep] if keep < len(d) else np.arange(len(d))
            sr, st = M._kabsch(moved[ch], reference[idx[ch]]); rotation = sr @ rotation; translation = sr @ translation + st
            err = float(np.mean(d[ch]))
            if math.isfinite(previous) and abs(previous - err) <= 1e-8 * max(previous, 1.0): break
            previous = err
        out.append((rotation, translation, M._symmetric_mean_distance(reference, candidate @ rotation.T + translation)))
    return sorted(out, key=lambda t: t[2])
def rim_axis(v):
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c); g = [abs(w[i]-w[(i+1)%3]) + abs(w[i]-w[(i+2)%3]) - abs(w[(i+1)%3]-w[(i+2)%3]) for i in range(3)]
    a = U[:, int(np.argmax(g))]; h = c @ a; r = np.linalg.norm(c - np.outer(h, a), axis=1)
    top = r[h > np.percentile(h, 85)].mean(); bot = r[h < np.percentile(h, 15)].mean(); return a if top > bot else -a
if (OUT / 'source').exists(): raise SystemExit('refusing to overwrite ' + str(OUT / 'source'))
src2 = OUT / 'source' / name_ep; shutil.copytree(SRC, src2, ignore=shutil.ignore_patterns('*.mp4', '__pycache__'))
epd2 = OUT / 'episode_dir' / name_ep; shutil.copytree(EPD, epd2)
if not (epd2 / 'stage2').exists(): shutil.copytree(R0 / 'episodes/episode_000032/stage2', epd2 / 'stage2')
roster = json.load(open(V8 / 'submission_kit/data/track_3_evaluation_objects.json'))['episodes'][str(EP)]
sub = pq.read_table(V8 / 'track3_submission.parquet', columns=['row_id', 'object_mesh_b64']).to_pandas()
z = np.load(SRC / 'object_poses.npz'); names = [str(x) for x in z['names']]; T0 = z['world_from_object'][0]
import coacd
settings = dict(threshold=0.01, max_convex_hull=128, preprocess_mode='auto', preprocess_resolution=100, resolution=2000, mcts_nodes=10, mcts_iterations=50, mcts_max_depth=3, merge=True, decimate=False, seed=44)
report = {}
for name in ('blue_cup',):
    slot = roster.index(name); rows = sub[sub.row_id.str.startswith(f'track_3/e{EP:06d}/') & sub.row_id.str.endswith(f'/o{slot}') & (sub.object_mesh_b64.fillna('').str.len() > 0)]
    cv, cf = M._decode_object_mesh(rows.object_mesh_b64.iloc[0]); cv = np.asarray(cv, float); cf = np.asarray(cf, np.int64)
    scan = trimesh.load(f'/data1/SRL_V2D/gt_best_per_object/{name}.ply', force='mesh', process=False); rv, rf = np.asarray(scan.vertices, float), np.asarray(scan.faces, np.int64)
    cc, cr = M._surface_moments(cv, cf); rc, rr = M._surface_moments(rv, rf); s = rr / cr; seed = 1_000_003 * EP + 101 * slot
    sols = align_all(M._sample_mesh_surface(rv, rf, 3000, seed) - rc, (M._sample_mesh_surface(cv, cf, 3000, seed) - cc) * s); best = sols[0]
    a_ours = rim_axis(cv); a_scan = rim_axis(rv); cands = []
    for i, (rot, off, res) in enumerate(sols):
        if res - best[2] > 0.001: continue
        cands.append(dict(index=i, residual_cm=res * 100, rim_dot=float((rot.T @ a_scan) @ a_ours), rot=rot, off=off))
    ok = [c for c in cands if c['rim_dot'] > 0.9] or cands; chosen = min(ok, key=lambda c: c['residual_cm'])
    rot, off = chosen['rot'], chosen['off']; c0 = cc + rot.T @ (-off) / s; Vb = c0 + (rv - rc) @ rot
    Tw = T0[names.index(name)]; old = trimesh.load(R0 / 'refinement_20261018/expanded_batch/cup32_height_candidate/scale_0.900/assets/blue_cup_visual.obj', force='mesh', process=False)   # the repaired (shortened) cup used in the scenes
    ob = float((np.asarray(old.vertices) @ Tw[:3, :3].T + Tw[:3, 3])[:, 2].min()); nb = float((Vb @ Tw[:3, :3].T + Tw[:3, 3])[:, 2].min())
    Vb = Vb + Tw[:3, :3].T @ np.array([0, 0, ob - nb]); mesh = trimesh.Trimesh(Vb, rf, process=False)
    parts = coacd.run_coacd(coacd.Mesh(Vb, rf), **settings); hulls = [trimesh.Trimesh(v, f, process=True).convex_hull for v, f in parts]
    vol = np.array([h.volume for h in hulls]); rho = 0.1 / vol.sum(); com = sum(h.volume * h.center_mass for h in hulls) / vol.sum(); I = np.zeros((3, 3))
    for h in hulls:
        m_i = rho * h.volume; d = h.center_mass - com; I += h.moment_inertia / h.volume * m_i + m_i * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    for root in (src2 / 'assets', epd2 / 'stage2/assets' / name):
        for f in (f'{name}_visual.obj',) + ((f'{name}.obj', f'{name}.ply') if root.name == 'assets' else ()):
            if f.endswith('.ply'): mesh.export(root / f)
            else: mesh.export(root / f)
        cd = root / f'{name}_convex'
        if cd.exists(): shutil.move(str(cd), str(root / f'{name}_convex_previous_ours'))
        cd.mkdir()
        for k, h in enumerate(hulls): h.export(cd / f'part_{k:03d}.obj')
    t = ET.parse(SRC / f'assets/{name}.urdf'); link = t.getroot().find('link'); inert = link.find('inertial')
    inert.find('origin').set('xyz', ' '.join(repr(float(x)) for x in com))
    for kk, ax in zip(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))): inert.find('inertia').set(kk, repr(float(I[ax])))
    for c in list(link.findall('collision')): link.remove(c)
    for k in range(len(hulls)):
        c = ET.SubElement(link, 'collision'); g = ET.SubElement(c, 'geometry'); ET.SubElement(g, 'mesh', filename=f'{name}_convex/part_{k:03d}.obj')
    ET.indent(t); t.write(src2 / f'assets/{name}.urdf', encoding='utf-8', xml_declaration=True); shutil.copy2(src2 / f'assets/{name}.urdf', OUT / f'{name}.urdf')
    report[name] = dict(slot=slot, scale_scan_over_ours=float(s), candidates=[{k: v for k, v in c.items() if k not in ('rot', 'off')} for c in cands], chosen_index=chosen['index'],
                        lift_world_z_cm=(ob - nb) * 100, convex_parts=len(hulls), inertia_com=com.tolist(), visual_sha256=sha(src2 / f'assets/{name}_visual.obj'))
    print(name, json.dumps({k: v for k, v in report[name].items() if k != 'candidates'}), 'cands', [(c['index'], round(c['residual_cm'], 2), round(c['rim_dot'], 2)) for c in cands], flush=True)
# manifest + metric geometry of the new episode dir point at the new source
m = json.loads((epd2 / 'input_manifest.json').read_text()); m['source'] = str(src2)
for o in m['objects']:
    nm = o['id']; o['source_urdf'] = str(src2 / f'assets/{nm}.urdf'); o['source_urdf_sha256'] = sha(src2 / f'assets/{nm}.urdf')
    o['meshes'] = [dict(source=str(src2 / f'assets/{nm}_visual.obj'), sha256=sha(src2 / f'assets/{nm}_visual.obj'), kind='visual')] + [
        dict(source=str(p), sha256=sha(p), kind='collision') for p in sorted((src2 / f'assets/{nm}_convex').glob('part_*.obj'))]
m['scan_cup_replacement'] = dict(note='blue_cup and beige_cup geometry replaced by the scan-derived meshes in our body frames; object poses unchanged', report=report)
(epd2 / 'input_manifest.json').write_text(json.dumps(m, indent=2))
sys.path.insert(0, str(R0 / 'code')); from evaluate_mp_sr import geometry_from_reference
g_old = json.loads((epd2 / 'metric_geometry.json').read_text()); g_new = geometry_from_reference(src2); g_old['objects'] = g_new['objects']
(epd2 / 'metric_geometry.json').write_text(json.dumps(g_old, indent=1))
(OUT / 'cup_scan_provenance.json').write_text(json.dumps(dict(report=report, source=str(src2), episode_dir=str(epd2), object_poses_sha256=sha(src2 / 'object_poses.npz'), original_object_poses_sha256=sha(SRC / 'object_poses.npz')), indent=1))
print('done; object poses identical:', sha(src2 / 'object_poses.npz') == sha(SRC / 'object_poses.npz'))

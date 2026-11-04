#!/usr/bin/env python3
"""Stream I (episode 30): build a complete physics world for episode 30 around a plastic_dish_rack VISUAL mesh of our own
reconstruction, with explicit lineage, and check it. Nothing scan-derived is read: hands, object poses, camera and the
mini_sweeper assets come from dexagent_remaining7_20261019/hawor_recon/episode_000030 (HaWoR hands + the user's final_recon_ours
ZIP objects); the rack visual mesh comes from --rack-visual; its collision parts are CoACD-decomposed from that mesh by the
unchanged repair_collision_assets.py (same CoACD settings as the shipped recon world's collision_library: threshold 0.01, 128 hulls, seed 44) and validated
against the sweeper reference poses; the episode dir is prepared by the unchanged prepare_seven.py; the physics scene is the
unchanged floating_runtime._freeze + calibrate_controller.controller_scene('kp5000_r200_d5').

Body frame convention (MUST hold for a new mesh): the rack mesh vertices are expressed in the rack body frame of
object_poses.npz (world_from_object of 'plastic_dish_rack'), metric metres, no scale, no visual origin offset (the URDF visual
origin is identity). Reference object poses are never changed. The current reconstruction mesh (sha256 fa3ade62...) defines that
frame; a new mesh is compared against it (bounds, centroid, rigid ICP offset) and the world is checked at frame 0 (penetrations,
3 s settle with the robot held still).

usage: build_world.py --rack-visual MESH.obj --workspace OUT_DIR [--no-track-correction] [--allow-reuse-parts] [--collision-cut]
                      [--threshold 0.01] [--max-hulls 128] [--skip-checks]
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, hashlib, json, shutil, sys, time, types
from pathlib import Path
import numpy as np

R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
HAWOR = R7 / 'hawor_recon/episode_000030'                       # HaWoR hands + ZIP objects (no scan anywhere)
REF_RACK = R7 / 'collision_recon/episode_000030/assets/plastic_dish_rack_visual.obj'   # fa3ade62..., defines the body frame
REF_PARTS = R7 / 'collision_library/episode_000030/plastic_dish_rack/parts'            # validated CoACD parts of REF_RACK
FORBIDDEN_ROOTS = [Path('/data1/SRL_V2D/gt_best_per_object'), R7 / 'rackscan_20261021']   # scan-derived geometry lives here
EP = 30; NAME = 'episode_000030'; OBJ = 'plastic_dish_rack'
sys.path[:0] = [str(R7 / 'code'), str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def mesh_frame_report(new_path):
    import trimesh
    old = trimesh.load(REF_RACK, force='mesh', process=False); new = trimesh.load(new_path, force='mesh', process=False)
    rep = dict(reference_mesh=str(REF_RACK), reference_sha256=sha(REF_RACK), new_mesh=str(new_path), new_sha256=sha(new_path),
               new_vertices=int(len(new.vertices)), new_faces=int(len(new.faces)), new_watertight=bool(new.is_watertight),
               old_bounds_m=old.bounds, new_bounds_m=new.bounds, old_extents_m=old.extents, new_extents_m=new.extents,
               old_centroid_m=old.vertices.mean(0), new_centroid_m=new.vertices.mean(0))
    rep['centroid_offset_cm'] = float(np.linalg.norm(rep['new_centroid_m'] - rep['old_centroid_m']) * 100)
    rep['extent_ratio'] = (new.extents / old.extents)
    if sha(new_path) != sha(REF_RACK):
        try:   # rigid ICP of the new surface onto the reference surface: a large offset means the mesh is not in the body frame
            a = new.sample(4000); b = old.sample(20000)
            T, _, cost = trimesh.registration.icp(a, b, max_iterations=60, reflection=False, scale=False)
            rot = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
            rep['icp_new_to_reference'] = dict(translation_cm=(T[:3, 3] * 100), rotation_deg=rot, mean_distance_cm=float(cost * 100))
            rep['frame_warning'] = bool(np.linalg.norm(T[:3, 3]) > .01 or rot > 5)
        except Exception as e: rep['icp_error'] = repr(e)
    else: rep['identical_to_reference_mesh'] = True; rep['frame_warning'] = False
    return rep


def stage_hawor_source(w, new_mesh):
    """Copy of the HaWoR/ZIP source with the rack visual replaced (bytes verbatim) and the coarse ZIP collision replaced by one
    convex hull of the new mesh (only used as the 'old' baseline inside the CoACD validation; not shipped)."""
    import trimesh
    dst = w / 'source_hawor' / NAME
    shutil.copytree(HAWOR, dst, ignore=shutil.ignore_patterns('archive_wilor', '*.mp4', '__pycache__', '*.pyc'))
    A = dst / 'assets'
    for f in (f'{OBJ}_visual.obj', f'{OBJ}.obj'): shutil.copyfile(new_mesh, A / f)
    m = trimesh.load(new_mesh, force='mesh', process=False)
    if sha(new_mesh) != sha(REF_RACK):
        m.export(A / f'{OBJ}.ply')
        shutil.rmtree(A / f'{OBJ}_convex'); (A / f'{OBJ}_convex').mkdir()
        m.convex_hull.export(A / f'{OBJ}_convex/part_000.obj')
        import xml.etree.ElementTree as ET
        tree = ET.parse(A / f'{OBJ}.urdf'); link = tree.getroot().find('link')
        for c in list(link.findall('collision')): link.remove(c)
        c = ET.SubElement(link, 'collision'); g = ET.SubElement(c, 'geometry'); ET.SubElement(g, 'mesh', filename=f'{OBJ}_convex/part_000.obj')
        ET.indent(tree); tree.write(A / f'{OBJ}.urdf', encoding='utf-8', xml_declaration=True)
    (dst / 'STREAM_I_SOURCE.txt').write_text(f'Copy of {HAWOR} with assets/{OBJ}_visual.obj and {OBJ}.obj replaced by {new_mesh} '
        f'(sha256 {sha(new_mesh)}); hands, camera, object poses, reference.parquet and mini_sweeper assets byte-identical to the source.\n')
    return dst


def coacd_parts(w, allow_reuse, threshold, max_hulls, new_mesh):
    import repair_collision_assets as RC
    a = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, object=OBJ, library=w / 'collision_library',
                              threshold=threshold, max_hulls=max_hulls, reuse_parts=None)
    if allow_reuse and sha(new_mesh) == sha(REF_RACK): a.reuse_parts = REF_PARTS   # identical mesh: validated identical parts
    RC.generate(a)
    m = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, library=w / 'collision_library', output_root=w / 'collision_recon')
    RC.merge(m)
    return w / 'collision_recon' / NAME, json.loads((w / 'collision_library' / NAME / OBJ / 'completed.json').read_text())


def correct_sweeper_track(src, dst):
    """Identical math to rackscan_20261021/ep30_v8/code/correct_sweeper_track.py (half-turn twin flip of the symmetric brush,
    2.5-3.2 s, mesh-centre trajectory unchanged, rack untouched), with explicit src/dst."""
    import trimesh, pyarrow as pa, pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation as R, Slerp
    shutil.copytree(src, dst)
    z = dict(np.load(src / 'object_poses.npz')); T = z['world_from_object'].copy(); names = [str(x) for x in z['names']]; i = names.index('mini_sweeper')
    V = np.asarray(trimesh.load(src / 'assets/mini_sweeper_visual.obj', force='mesh', process=False).vertices); c = V.mean(0); w_, U = np.linalg.eigh((V - c).T @ (V - c)); L = U[:, 2]
    F = R.from_rotvec(np.pi * L).as_matrix(); t = np.arange(len(T)) / 20.; t0, t1 = 2.5, 3.2
    cw = T[:, i, :3, :3] @ c + T[:, i, :3, 3]
    Rn = T[:, i, :3, :3].copy(); k0, k1 = int(t0 * 20), int(t1 * 20)
    Rn[k1:] = T[k1:, i, :3, :3] @ F
    sl = Slerp([t0, t1], R.from_matrix([T[k0, i, :3, :3], Rn[k1]]))
    for k in range(k0 + 1, k1): u = (t[k] - t0) / (t1 - t0); u = u * u * (3 - 2 * u); Rn[k] = sl(t0 + u * (t1 - t0)).as_matrix()
    T[:, i, :3, :3] = Rn; T[:, i, :3, 3] = cw - np.einsum('kij,j->ki', Rn, c)
    z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
    tr = dict(np.load(src / 'mini_sweeper_trajectory.npz')); Two = tr['T_world_obj'].copy(); Two[:, :3, :3] = Rn; Two[:, :3, 3] = T[:, i, :3, 3]
    tr['T_world_obj'] = Two; tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ Two; np.savez_compressed(dst / 'mini_sweeper_trajectory.npz', **tr)
    tab = pq.read_table(src / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names']); j = bn.index('mini_sweeper')
    pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz']); pos[:, j] = T[:, i, :3, 3]; qq = R.from_matrix(Rn).as_quat(); q[:, j] = qq[:, [3, 0, 1, 2]]
    if bn[0] == row['object_name']:
        row['object_root_position'] = pos[:, 0].tolist(); rv = R.from_matrix(Rn).as_rotvec() if j == 0 else None
        if rv is not None: row['object_root_axis_angle'] = rv.tolist()
    row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist()
    pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), dst / 'reference.parquet')
    rel = np.degrees((R.from_matrix(Rn) * R.from_matrix(Rn[40]).inv()).magnitude())
    info = dict(episode=30, object='mini_sweeper', window_s=[t0, t1], twin_axis_body=L.tolist(), twin_centre_body=c.tolist(), centre_trajectory_unchanged=True,
        evidence=dict(human_right_wrist_turn_deg_2_to_4s=30, original_track_turn_deg=178, twin_vs_rigid_hand_hold_deg='18-30 for 3.2-12 s', original_vs_rigid_hand_deg='150-166',
                      rgb_overlay='rackscan_20261021/ep30_v8/analysis/overlay_a.jpg: rot 0 and rot 180 both match the silhouette at 1 s and 4 s (symmetric object); +-90 do not',
                      first_applied='rackscan_20261021/ep30_v8/code/correct_sweeper_track.py (v8..v17 ship it); re-applied here to the recon-rack source with identical math'),
        corrected_turn_from_2s_deg={f'{s}': float(rel[int(s * 20)]) for s in (3.0, 3.4, 4.0, 8.0, 12.0)},
        original_object_poses_sha256=sha(src / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
        original_reference_parquet_sha256=sha(src / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'),
        rack_columns_unchanged=bool(np.array_equal(np.load(src / 'object_poses.npz')['world_from_object'][:, names.index(OBJ)], T[:, names.index(OBJ)])))
    (dst / 'sweeper_track_correction.json').write_text(json.dumps(info, indent=2))
    return info


def override_poses(src, dst, npz):
    """Replace the reference object poses by an externally prepared track (e.g. the compose_eval track expressed in our world,
    phase 6). Writes object_poses.npz, <obj>_trajectory.npz and reference.parquet object columns; everything else byte-identical.
    The record keeps the corrected-track wrapper's keys (sweeper_track_correction.json) so the unchanged staging wrapper accepts it."""
    import pyarrow as pa, pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation as R
    shutil.copytree(src, dst)
    o = np.load(npz); z = dict(np.load(src / 'object_poses.npz')); names = [str(x) for x in z['names']]
    if [str(x) for x in o['names']] != names or o['world_from_object'].shape != z['world_from_object'].shape: raise ValueError('override poses must match names/shape')
    T = np.asarray(o['world_from_object'], float); z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
    for i, n in enumerate(names):
        f = src / f'{n}_trajectory.npz'
        if f.exists():
            tr = dict(np.load(f)); tr['T_world_obj'] = T[:, i].copy(); tr['T_rgb_obj'] = np.linalg.inv(z['world_from_rgb']) @ T[:, i]; np.savez_compressed(dst / f.name, **tr)
    tab = pq.read_table(src / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names'])
    pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz'])
    for i, n in enumerate(names):
        if n in bn:
            j = bn.index(n); pos[:, j] = T[:, i, :3, 3]; qq = R.from_matrix(T[:, i, :3, :3]).as_quat(); q[:, j] = qq[:, [3, 0, 1, 2]]
    if bn[0] == row['object_name']:
        k = names.index(bn[0]); row['object_root_position'] = pos[:, 0].tolist(); row['object_root_axis_angle'] = R.from_matrix(T[:, k, :3, :3]).as_rotvec().tolist()
    row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist()
    pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), dst / 'reference.parquet')
    lin = Path(str(npz).replace('.npz', '_lineage.json'))
    info = dict(episode=EP, kind='reference object track replaced by an external track (stream I phase 6: compose_eval target)', override_npz=str(npz), override_sha256=sha(npz),
                lineage=json.loads(lin.read_text()) if lin.exists() else None,
                original_object_poses_sha256=sha(src / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
                original_reference_parquet_sha256=sha(src / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'))
    (dst / 'sweeper_track_correction.json').write_text(json.dumps(info, indent=2))
    return info


def prepare(w, source_root):
    import prepare_seven as PS
    PS.BASE = w
    res = PS.prepare_episode(EP, source_root)
    shutil.copyfile(R7 / 'input_manifest.json', w / 'input_manifest.json')   # the user ZIP manifest (staging bases need it)
    return res


def build_scene(w, epdir):
    from floating_runtime import _freeze
    from calibrate_controller import controller_scene
    sd = w / 'scene'; sd.mkdir()
    _, frozen = _freeze(epdir / 'stage2/scene.xml', sd)
    ctl = controller_scene(frozen, sd / 'controller_scene.xml', 'kp5000_r200_d5')
    dump(sd / 'controller_scene.json', dict(source=str(epdir / 'stage2/scene.xml'), frozen=str(frozen), controller=ctl, variant='kp5000_r200_d5',
                                           builder='floating_runtime._freeze + calibrate_controller.controller_scene (run_seven_mpc baseline recipe)'))
    return sd / 'controller_scene.xml'


def mesh_manifest(w, epdir, scene):
    import xml.etree.ElementTree as ET
    forb = {}
    for root in FORBIDDEN_ROOTS:
        for p in root.rglob('*'):
            if p.suffix.lower() in ('.obj', '.ply', '.stl', '.glb') and p.is_file() and 'rack' in p.name.lower(): forb[sha(p)] = str(p)
    rows = []
    for m in ET.parse(scene).getroot().iter('mesh'):
        f = m.get('file')
        if not f or m.get('name', '').startswith('robot_'): continue
        rows.append(dict(name=m.get('name'), file=f, bytes=Path(f).stat().st_size, sha256=sha(f), scan_hash_match=forb.get(sha(f))))
    bad = [r for r in rows if r['scan_hash_match']]
    return dict(scene=str(scene), forbidden_roots=[str(r) for r in FORBIDDEN_ROOTS], forbidden_rack_hashes_checked=len(forb), object_meshes=rows, scan_matches=bad, clean=not bad)


def frame0_and_settle(w, epdir, scene):
    import mujoco
    from tool_library.common import tracking as TR
    from floating_runtime import execute_reference
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    def bodyname(g): return m.body(int(m.geom_bodyid[g])).name
    pens = []
    for c in d.contact[:d.ncon]:
        b1, b2 = bodyname(c.geom1), bodyname(c.geom2)
        if any(k in (b1, b2) for k in ('mini_sweeper', OBJ)):
            pens.append(dict(geom1=m.geom(c.geom1).name, geom2=m.geom(c.geom2).name, body1=b1, body2=b2, dist_mm=float(c.dist * 1000)))
    pens.sort(key=lambda r: r['dist_mm'])
    summ = {}
    for r in pens:
        key = '|'.join(sorted({r['body1'], r['body2']})); summ.setdefault(key, []).append(r['dist_mm'])
    summary = {k: dict(contacts=len(v), deepest_mm=min(v), count_below_minus1mm=int(sum(x < -1 for x in v))) for k, v in summ.items()}
    # settle: robot held at its frame-0 reference pose for 3 s, objects free -> drift of each object vs its frame-0 reference
    ref = TR.load_reference(epdir / 'tracking_reference.npz', m); q = ref['qpos'][:61].copy(); q[:] = q[0]
    sd = w / 'checks/settle'; sd.mkdir(parents=True)
    p = TR.save_reference(sd / 'reference.npz', m, q, ref['labels'][:61], ref['fps'], ref['segment'][:61], ref['segments'], source='world check: robot frozen at frame 0, objects free', floating_runtime=True, no_retiming=True)
    ex = execute_reference(str(scene), p, sd / 'execution')
    tr = np.load(ex['trajectory']); names = [str(x) for x in tr['object_names']]; tl = {}
    for i, n in enumerate(names):
        p0 = tr['object_positions'][0, i]; q0 = tr['object_quaternions_wxyz'][0, i]
        from scipy.spatial.transform import Rotation as R
        r0 = R.from_quat(q0[[1, 2, 3, 0]])
        tl[n] = {f'{k/20:.1f}s': dict(cm=float(np.linalg.norm(tr['object_positions'][k, i] - p0) * 100),
                                  deg=float(np.degrees((R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]) * r0.inv()).magnitude())))
                 for k in range(0, 61, 10)}
    return dict(frame0_contacts=summary, frame0_contacts_detail=pens[:40], settle_drift_robot_frozen=tl, settle_execution=ex['scene'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rack-visual', type=Path, required=True); ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--no-track-correction', action='store_true', help='keep the ZIP sweeper track (178 deg twin flip at 2.5-3.2 s)')
    ap.add_argument('--allow-reuse-parts', action='store_true', help='if the mesh is byte-identical to the shipped recon rack, reuse its validated CoACD parts')
    ap.add_argument('--threshold', type=float, default=.01); ap.add_argument('--max-hulls', type=int, default=128); ap.add_argument('--skip-checks', action='store_true')
    ap.add_argument('--pose-override', type=Path, help='npz with world_from_object (N,objects,4,4) + names in our world; replaces the reference track (no twin-flip correction)')
    ap.add_argument('--collision-cut', action='store_true', help='slice rack collision parts that interpenetrate the sweeper at its frame-0 reference pose (cut_rack_collision.py)')
    ap.add_argument('--cut-frames', type=int, nargs='+', default=[0], help='reference frames whose sweeper pose the rack collision must not overlap'); ap.add_argument('--cut-tol', type=float, default=.0005); ap.add_argument('--cut-margin', type=float, default=.0005)
    a = ap.parse_args(); w = a.workspace.resolve(); new_mesh = a.rack_visual.resolve()
    if w.exists(): raise SystemExit(f'workspace exists, never overwrite: {w}')
    w.mkdir(parents=True); lineage = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), args=vars(a), started=time.strftime('%F %T'))
    log('mesh frame report'); lineage['rack_mesh'] = mesh_frame_report(new_mesh); dump(w / 'lineage.json', lineage)
    if lineage['rack_mesh'].get('frame_warning'): log('WARNING: the new rack mesh is offset from the reference body frame', lineage['rack_mesh'].get('icp_new_to_reference'))
    log('stage source'); src = stage_hawor_source(w, new_mesh)
    log('coacd'); recon, report = coacd_parts(w, a.allow_reuse_parts, a.threshold, a.max_hulls, new_mesh)
    lineage['collision'] = dict(parts=len(report['convex_parts']), generation=report['generation'], validation_passed=report['validation']['passed'],
                                validation=str(w / 'collision_library' / NAME / OBJ / 'validation.json'), merged=str(recon))
    if a.collision_cut:
        import subprocess
        log('collision cut'); out = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'cut_rack_collision.py'), str(recon), '--tol', str(a.cut_tol), '--margin', str(a.cut_margin), '--frame', *[str(f) for f in a.cut_frames]] + (['--poses', str(a.pose_override.resolve())] if a.pose_override else []), check=True, capture_output=True, text=True).stdout
        lineage['collision_cut'] = json.loads(out.strip().splitlines()[-1]); lineage['collision_cut']['report'] = str(recon / 'rack_collision_cut.json'); dump(w / 'lineage.json', lineage)
    source_root = w / 'collision_recon'
    if a.pose_override:
        log('pose override'); info = override_poses(recon, w / 'collision_recon_corrected' / NAME, a.pose_override.resolve()); lineage['pose_override'] = {k: v for k, v in info.items() if k != 'lineage'}; source_root = w / 'collision_recon_corrected'
    elif not a.no_track_correction:
        log('track correction'); info = correct_sweeper_track(recon, w / 'collision_recon_corrected' / NAME); lineage['track_correction'] = info; source_root = w / 'collision_recon_corrected'
    dump(w / 'lineage.json', lineage)
    log('prepare episode dir'); res = prepare(w, source_root); epdir = Path(res['input_dir']); lineage['prepare'] = res
    log('physics scene'); scene = build_scene(w, epdir); lineage['scene'] = str(scene)
    log('mesh manifest'); mm = mesh_manifest(w, epdir, scene); dump(w / 'world_meshes.json', mm); lineage['meshes_clean'] = mm['clean']
    if not mm['clean']: raise SystemExit('scan-derived mesh detected: ' + json.dumps(mm['scan_matches']))
    if not a.skip_checks:
        log('frame-0 + settle checks'); chk = frame0_and_settle(w, epdir, scene); dump(w / 'world_check.json', chk)
        lineage['checks'] = dict(frame0=chk['frame0_contacts'], settle=chk['settle_drift_robot_frozen'])
    lineage['finished'] = time.strftime('%F %T'); lineage['episode_dir'] = str(epdir); dump(w / 'lineage.json', lineage)
    log('DONE', json.dumps(dict(episode_dir=str(epdir), scene=str(scene), parts=lineage['collision']['parts'], clean=mm['clean'])))


if __name__ == '__main__': main()

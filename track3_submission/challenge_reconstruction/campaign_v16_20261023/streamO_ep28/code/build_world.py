#!/usr/bin/env python3
"""Stream O (episode 28): build a complete physics world for episode 28 around a plastic_dish_rack VISUAL mesh of our own
reconstruction, with explicit lineage, and check it. Generalised from streamI_ep30/code/build_world.py (episode 30); the
sweeper track correction of episode 30 does not exist here (episode 28 has no track correction).

Nothing scan-derived is read: hands, object poses, camera and the white_pot assets come from
dexagent_remaining7_20261019/hawor_recon/episode_000028 (HaWoR hands + the user's final_recon_ours ZIP objects); the rack visual
mesh comes from --rack-visual; its collision parts are CoACD-decomposed from that mesh by the unchanged repair_collision_assets.py
(same CoACD settings as the shipped recon world's collision_library: threshold 0.01, 128 hulls, seed 44) and validated against
the white_pot reference poses; the white_pot collision parts are the validated 64-part CoACD decomposition of OUR pot mesh
(collision_library/episode_000028/white_pot, threshold 0.03 / 64 hulls / seed 44, reused byte-identically); the episode dir is
prepared by the unchanged prepare_seven.py; the physics scene is the unchanged floating_runtime._freeze +
calibrate_controller.controller_scene('kp5000_r200_d5') (run_seven_mpc baseline recipe = the pre-scan teacher's scene).

Body frame convention (MUST hold for a new mesh): the rack mesh vertices are expressed in the rack body frame of
object_poses.npz (world_from_object of 'plastic_dish_rack'), metric metres, no scale, no visual origin offset (the URDF visual
origin is identity). Reference object poses are never changed. The current reconstruction mesh (sha256 d4bddc30...) defines that
frame; a new mesh is compared against it (bounds, centroid, rigid ICP offset) and the world is checked at frame 0 (penetrations,
3 s settle with the robot held still).

usage: build_world.py --rack-visual MESH.obj --workspace OUT_DIR [--allow-reuse-parts] [--threshold 0.01] [--max-hulls 128] [--skip-checks]
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, hashlib, json, shutil, sys, time, types
from pathlib import Path
import numpy as np

R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
EP = 28; NAME = 'episode_000028'; OBJ = 'plastic_dish_rack'; MOVING = 'white_pot'
HAWOR = R7 / 'hawor_recon' / NAME                                                  # HaWoR hands + ZIP objects (no scan anywhere)
REF_RACK = R7 / 'collision_recon' / NAME / 'assets/plastic_dish_rack_visual.obj'   # d4bddc30..., defines the body frame
REF_PARTS = R7 / 'collision_library' / NAME / OBJ / 'parts'                        # validated CoACD parts of REF_RACK (128)
POT_PARTS = R7 / 'collision_library' / NAME / MOVING / 'parts'                     # validated CoACD parts of our pot (64)
POT_SETTINGS = dict(threshold=.03, max_hulls=64)
FORBIDDEN_ROOTS = [Path('/data1/SRL_V2D/gt_best_per_object'), R7 / 'rackscan_20261021']   # scan-derived geometry lives here
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
    (dst / 'STREAM_O_SOURCE.txt').write_text(f'Copy of {HAWOR} with assets/{OBJ}_visual.obj and {OBJ}.obj replaced by {new_mesh} '
        f'(sha256 {sha(new_mesh)}); hands, camera, object poses, reference.parquet and {MOVING} assets byte-identical to the source.\n')
    return dst


def coacd_parts(w, allow_reuse, threshold, max_hulls, new_mesh):
    import repair_collision_assets as RC
    a = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, object=OBJ, library=w / 'collision_library',
                              threshold=threshold, max_hulls=max_hulls, reuse_parts=None)
    if allow_reuse and sha(new_mesh) == sha(REF_RACK): a.reuse_parts = REF_PARTS   # identical mesh: validated identical parts
    RC.generate(a)
    # the pot mesh never changes: reuse its validated CoACD parts byte-identically (repair_collision_assets re-validates them)
    b = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, object=MOVING, library=w / 'collision_library',
                              threshold=POT_SETTINGS['threshold'], max_hulls=POT_SETTINGS['max_hulls'], reuse_parts=POT_PARTS)
    RC.generate(b)
    m = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, library=w / 'collision_library', output_root=w / 'collision_recon')
    RC.merge(m)
    reports = {o: json.loads((w / 'collision_library' / NAME / o / 'completed.json').read_text()) for o in (OBJ, MOVING)}
    return w / 'collision_recon' / NAME, reports


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
    """Every object mesh the physics scene references, with sha256, checked against (a) every mesh file under gt_best_per_object
    and (b) every rack mesh under rackscan_20261021 (that tree also holds copies of OUR pot parts, so only rack files count there)."""
    import xml.etree.ElementTree as ET
    forb = {}
    for root in FORBIDDEN_ROOTS:
        for p in root.rglob('*'):
            if p.suffix.lower() in ('.obj', '.ply', '.stl', '.glb') and p.is_file() and (root.name == 'gt_best_per_object' or 'rack' in p.name.lower()):
                forb[sha(p)] = str(p)
    rows = []
    for m in ET.parse(scene).getroot().iter('mesh'):
        f = m.get('file')
        if not f or m.get('name', '').startswith('robot_'): continue
        rows.append(dict(name=m.get('name'), file=f, bytes=Path(f).stat().st_size, sha256=sha(f), scan_hash_match=forb.get(sha(f))))
    bad = [r for r in rows if r['scan_hash_match']]
    ref = {o: {sha(p) for p in sorted(d.glob('part_*.obj'))} for o, d in ((OBJ, REF_PARTS), (MOVING, POT_PARTS))}
    same = {o: sum(r['sha256'] in ref[o] for r in rows if f'/{o}/' in r['file'] and 'part_' in r['file']) for o in ref}
    return dict(scene=str(scene), forbidden_roots=[str(r) for r in FORBIDDEN_ROOTS], forbidden_hashes_checked=len(forb), object_meshes=rows,
                scan_matches=bad, clean=not bad, parts_identical_to_shipped_recon_library=same)


def frame0_and_settle(w, epdir, scene):
    import mujoco
    from tool_library.common import tracking as TR
    from floating_runtime import execute_reference
    from scipy.spatial.transform import Rotation as R
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    def bodyname(g): return m.body(int(m.geom_bodyid[g])).name
    pens = []
    for c in d.contact[:d.ncon]:
        b1, b2 = bodyname(c.geom1), bodyname(c.geom2)
        if any(k in (b1, b2) for k in (MOVING, OBJ)):
            pens.append(dict(geom1=m.geom(c.geom1).name, geom2=m.geom(c.geom2).name, body1=b1, body2=b2, dist_mm=float(c.dist * 1000)))
    pens.sort(key=lambda r: r['dist_mm'])
    summ = {}
    for r in pens:
        key = '|'.join(sorted({r['body1'], r['body2']})); summ.setdefault(key, []).append(r['dist_mm'])
    summary = {k: dict(contacts=len(v), deepest_mm=min(v), count_below_minus1mm=int(sum(x < -1 for x in v))) for k, v in summ.items()}
    # reference-pose penetration timeline of the moving object against the rack / table (kinematic, every 0.5 s)
    ref = TR.load_reference(epdir / 'tracking_reference.npz', m); Q = ref['qpos']; tl_pen = {}
    for k in range(0, len(Q), 10):
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d); worst = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = bodyname(c.geom1), bodyname(c.geom2)
            if MOVING in (b1, b2):
                o = b1 if b2 == MOVING else b2; worst[o] = min(worst.get(o, 0.), float(c.dist * 1000))
        tl_pen[f'{k/20:.1f}s'] = worst
    # settle: robot held at its frame-0 reference pose for 3 s, objects free -> drift of each object vs its frame-0 reference
    q = Q[:61].copy(); q[:] = q[0]
    sd = w / 'checks/settle'; sd.mkdir(parents=True)
    p = TR.save_reference(sd / 'reference.npz', m, q, ref['labels'][:61], ref['fps'], ref['segment'][:61], ref['segments'], source='world check: robot frozen at frame 0, objects free', floating_runtime=True, no_retiming=True)
    ex = execute_reference(str(scene), p, sd / 'execution')
    tr = np.load(ex['trajectory']); names = [str(x) for x in tr['object_names']]; tl = {}
    for i, n in enumerate(names):
        p0 = tr['object_positions'][0, i]; q0 = tr['object_quaternions_wxyz'][0, i]
        r0 = R.from_quat(q0[[1, 2, 3, 0]])
        tl[n] = {f'{k/20:.1f}s': dict(cm=float(np.linalg.norm(tr['object_positions'][k, i] - p0) * 100),
                                  deg=float(np.degrees((R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]) * r0.inv()).magnitude())))
                 for k in range(0, 61, 10)}
    return dict(frame0_contacts=summary, frame0_contacts_detail=pens[:40], reference_pose_penetration_mm_timeline=tl_pen, settle_drift_robot_frozen=tl, settle_execution=ex['scene'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rack-visual', type=Path, required=True); ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--allow-reuse-parts', action='store_true', help='if the mesh is byte-identical to the shipped recon rack, reuse its validated CoACD parts')
    ap.add_argument('--threshold', type=float, default=.01); ap.add_argument('--max-hulls', type=int, default=128); ap.add_argument('--skip-checks', action='store_true')
    a = ap.parse_args(); w = a.workspace.resolve(); new_mesh = a.rack_visual.resolve()
    if w.exists(): raise SystemExit(f'workspace exists, never overwrite: {w}')
    w.mkdir(parents=True); lineage = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), args=vars(a), started=time.strftime('%F %T'), episode=EP)
    log('mesh frame report'); lineage['rack_mesh'] = mesh_frame_report(new_mesh); dump(w / 'lineage.json', lineage)
    if lineage['rack_mesh'].get('frame_warning'): log('WARNING: the new rack mesh is offset from the reference body frame', lineage['rack_mesh'].get('icp_new_to_reference'))
    log('stage source'); src = stage_hawor_source(w, new_mesh)
    log('coacd'); recon, reports = coacd_parts(w, a.allow_reuse_parts, a.threshold, a.max_hulls, new_mesh)
    lineage['collision'] = {o: dict(parts=len(r['convex_parts']), generation=r['generation'], validation_passed=r['validation']['passed'],
                                    validation=str(w / 'collision_library' / NAME / o / 'validation.json'), source_visual_sha256=r['source_visual_sha256']) for o, r in reports.items()}
    lineage['collision']['merged'] = str(recon)
    dump(w / 'lineage.json', lineage)
    log('prepare episode dir'); res = prepare(w, w / 'collision_recon'); epdir = Path(res['input_dir']); lineage['prepare'] = res
    log('physics scene'); scene = build_scene(w, epdir); lineage['scene'] = str(scene)
    log('mesh manifest'); mm = mesh_manifest(w, epdir, scene); dump(w / 'world_meshes.json', mm); lineage['meshes_clean'] = mm['clean']
    if not mm['clean']: raise SystemExit('scan-derived mesh detected: ' + json.dumps(mm['scan_matches']))
    if not a.skip_checks:
        log('frame-0 + settle checks'); chk = frame0_and_settle(w, epdir, scene); dump(w / 'world_check.json', chk)
        lineage['checks'] = dict(frame0=chk['frame0_contacts'], settle=chk['settle_drift_robot_frozen'])
    lineage['finished'] = time.strftime('%F %T'); lineage['episode_dir'] = str(epdir); dump(w / 'lineage.json', lineage)
    log('DONE', json.dumps(dict(episode_dir=str(epdir), scene=str(scene), parts={o: v['parts'] for o, v in lineage['collision'].items() if isinstance(v, dict)}, clean=mm['clean'])))


if __name__ == '__main__': main()

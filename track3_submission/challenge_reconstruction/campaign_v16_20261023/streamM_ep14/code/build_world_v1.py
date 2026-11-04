#!/usr/bin/env python3
"""Stream M (episode 14, blue_cup into the plastic_dish_rack): build a complete physics world for episode 14 around a
plastic_dish_rack VISUAL mesh of OUR OWN reconstruction (default: the previously reconstructed rack ccc315ab3efa.., 702692
bytes, = final_recon/episode_000014/assets/plastic_dish_rack.obj), with explicit lineage, and check it. Nothing scan-derived is
read: hands, object poses, camera and the blue_cup assets come from dexagent_remaining7_20261019/hawor_stereo_recon_v2/
episode_000014 (HaWoR hands aligned with calibrated stereo = the hand source of the pre-scan world collision_recon/episode_000014
-> physics_collision_v2/episodes/episode_000014; hawor_recon/episode_000014 holds the pre-stereo hands and differs); the rack
visual mesh comes from --rack-visual; its collision parts are CoACD-decomposed from that mesh by the unchanged
repair_collision_assets.py (same settings as the shipped pre-scan collision_library: threshold 0.01, 128 hulls, seed 44), or, when
the mesh is byte-identical to the reference rack and --allow-reuse-parts is given, the 128 validated pre-scan parts are reused
(byte-identical); the blue_cup collision is always the validated 42-part pre-scan library decomposition of our cup mesh
(REPAIRS[14] = rack + cup in repair_collision_assets.py). The episode dir is prepared by the unchanged prepare_seven.py; the
physics scene is the unchanged floating_runtime._freeze + calibrate_controller.controller_scene('kp5000_r200_d5').

Body frame convention (MUST hold for a new mesh): the rack mesh vertices are expressed in the rack body frame of
object_poses.npz (world_from_object of 'plastic_dish_rack'), metric metres, no scale, no visual origin offset. Reference object
poses are never changed. The reference mesh (sha256 ccc315ab...) defines that frame; a new mesh is compared against it (bounds,
centroid, rigid ICP offset) and the world is checked at frame 0 and at the set-down pose (penetrations, 3 s settle with the
robot held still at frame 0, cup dropped into the rack at its final reference pose).

usage: build_world.py --rack-visual MESH.obj --workspace OUT_DIR [--allow-reuse-parts] [--threshold 0.01] [--max-hulls 128] [--skip-checks]
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, hashlib, json, shutil, sys, time, types
from pathlib import Path
import numpy as np

R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
EP = 14; NAME = 'episode_000014'; OBJ = 'plastic_dish_rack'; CUP = 'blue_cup'
HAWOR = R7 / 'hawor_stereo_recon_v2' / NAME                     # HaWoR hands (stereo-aligned) + ZIP objects; the pre-scan world's source
REF_RACK = R7 / 'collision_recon' / NAME / 'assets/plastic_dish_rack_visual.obj'   # ccc315ab..., defines the body frame
REF_PARTS = R7 / 'collision_library' / NAME / OBJ / 'parts'                         # 128 validated CoACD parts of REF_RACK (2026-10-19, pre-scan)
CUP_PARTS = R7 / 'collision_library' / NAME / CUP / 'parts'                         # 42 validated CoACD parts of our cup 78bd065c...
PRESCAN_EPDIR = R7 / 'physics_collision_v2/episodes' / NAME
FORBIDDEN_ROOTS = [Path('/data1/SRL_V2D/gt_best_per_object'), R7 / 'rackscan_20261021']   # scan-derived geometry lives here
sys.path[:0] = [str(R7 / 'code'), str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def mesh_frame_report(new_path):
    import trimesh
    old = trimesh.load(REF_RACK, force='mesh', process=False); new = trimesh.load(new_path, force='mesh', process=False)
    rep = dict(reference_mesh=str(REF_RACK), reference_sha256=sha(REF_RACK), new_mesh=str(new_path), new_sha256=sha(new_path), new_bytes=Path(new_path).stat().st_size,
               new_vertices=int(len(new.vertices)), new_faces=int(len(new.faces)), new_watertight=bool(new.is_watertight),
               old_bounds_m=old.bounds, new_bounds_m=new.bounds, old_extents_m=old.extents, new_extents_m=new.extents,
               old_centroid_m=old.vertices.mean(0), new_centroid_m=new.vertices.mean(0))
    rep['centroid_offset_cm'] = float(np.linalg.norm(rep['new_centroid_m'] - rep['old_centroid_m']) * 100)
    rep['extent_ratio'] = (new.extents / old.extents)
    if sha(new_path) != sha(REF_RACK):
        try:
            a = new.sample(4000); b = old.sample(20000)
            T, _, cost = trimesh.registration.icp(a, b, max_iterations=60, reflection=False, scale=False)
            rot = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
            rep['icp_new_to_reference'] = dict(translation_cm=(T[:3, 3] * 100), rotation_deg=rot, mean_distance_cm=float(cost * 100))
            rep['frame_warning'] = bool(np.linalg.norm(T[:3, 3]) > .01 or rot > 5)
        except Exception as e: rep['icp_error'] = repr(e)
    else: rep['identical_to_reference_mesh'] = True; rep['frame_warning'] = False
    return rep


def stage_hawor_source(w, new_mesh):
    """Copy of the HaWoR(stereo)/ZIP source with the rack visual replaced (bytes verbatim; identical for the default mesh) and,
    for a different mesh, the coarse ZIP rack collision replaced by one convex hull of the new mesh (only the 'old' baseline of
    the CoACD validation; never shipped)."""
    import trimesh
    dst = w / 'source_hawor' / NAME
    shutil.copytree(HAWOR, dst, ignore=shutil.ignore_patterns('archive_wilor', 'archive_before_stereo', '*.mp4', '__pycache__', '*.pyc'))
    A = dst / 'assets'
    for f in (f'{OBJ}_visual.obj', f'{OBJ}.obj'): shutil.copyfile(new_mesh, A / f)
    if sha(new_mesh) != sha(REF_RACK):
        m = trimesh.load(new_mesh, force='mesh', process=False); m.export(A / f'{OBJ}.ply')
        shutil.rmtree(A / f'{OBJ}_convex'); (A / f'{OBJ}_convex').mkdir()
        m.convex_hull.export(A / f'{OBJ}_convex/part_000.obj')
        import xml.etree.ElementTree as ET
        tree = ET.parse(A / f'{OBJ}.urdf'); link = tree.getroot().find('link')
        for c in list(link.findall('collision')): link.remove(c)
        c = ET.SubElement(link, 'collision'); g = ET.SubElement(c, 'geometry'); ET.SubElement(g, 'mesh', filename=f'{OBJ}_convex/part_000.obj')
        ET.indent(tree); tree.write(A / f'{OBJ}.urdf', encoding='utf-8', xml_declaration=True)
    (dst / 'STREAM_M_SOURCE.txt').write_text(f'Copy of {HAWOR} with assets/{OBJ}_visual.obj and {OBJ}.obj replaced by {new_mesh} '
        f'(sha256 {sha(new_mesh)}); hands, camera, object poses, reference.parquet and blue_cup assets byte-identical to the source.\n')
    return dst


def coacd_parts(w, allow_reuse, threshold, max_hulls, new_mesh):
    import repair_collision_assets as RC
    identical = sha(new_mesh) == sha(REF_RACK)
    a = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, object=OBJ, library=w / 'collision_library',
                              threshold=threshold, max_hulls=max_hulls, reuse_parts=REF_PARTS if (allow_reuse and identical) else None)
    RC.generate(a)
    c = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, object=CUP, library=w / 'collision_library',
                              threshold=threshold, max_hulls=max_hulls, reuse_parts=CUP_PARTS)   # the cup never changes: validated pre-scan parts
    RC.generate(c)
    m = types.SimpleNamespace(source_root=w / 'source_hawor', episode=EP, library=w / 'collision_library', output_root=w / 'collision_recon')
    RC.merge(m)
    reports = {o: json.loads((w / 'collision_library' / NAME / o / 'completed.json').read_text()) for o in (OBJ, CUP)}
    ident = {o: dict(parts=len(r['convex_parts']), identical_to_prescan_library=all(sha(p['path']) == sha(ref / Path(p['path']).name) for p in r['convex_parts']) if (ref / 'part_000.obj').exists() else None)
             for (o, r), ref in zip(reports.items(), (REF_PARTS, CUP_PARTS))}
    return w / 'collision_recon' / NAME, reports, ident


def prepare(w, source_root):
    import prepare_seven as PS
    PS.BASE = w
    res = PS.prepare_episode(EP, source_root)
    shutil.copyfile(R7 / 'input_manifest.json', w / 'input_manifest.json')
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


RECON_ROOTS = [R7 / 'collision_recon' / NAME, R7 / 'collision_library' / NAME, R7 / 'hawor_stereo_recon_v2' / NAME, R7 / 'hawor_recon' / NAME,
               R7 / 'source/final_recon_ours' / NAME]   # pre-scan (2026-10-19) reconstruction sources of episode 14: the lineage every mesh must trace to


def mesh_manifest(w, epdir, scene):
    """Every object mesh referenced by the physics scene, hashed. Each must be byte-identical to a file under the pre-scan
    reconstruction roots (lineage), and must not be a SCAN-ONLY file: the sha256 of every mesh file under the forbidden scan roots
    (any object, visual or collision part) minus the hashes that also exist under the recon roots (the scan worlds kept copies of
    our cup and our cup parts in their adapter_sources; those are ours, not scan-derived)."""
    import xml.etree.ElementTree as ET
    def hashes(roots):
        h = {}
        for root in roots:
            for p in root.rglob('*'):
                if p.suffix.lower() in ('.obj', '.ply', '.stl', '.glb') and p.is_file(): h.setdefault(sha(p), str(p))
        return h
    recon = hashes(RECON_ROOTS); scan_all = hashes(FORBIDDEN_ROOTS); scan_only = {k: v for k, v in scan_all.items() if k not in recon}
    rows = []
    for m in ET.parse(scene).getroot().iter('mesh'):
        f = m.get('file')
        if not f or m.get('name', '').startswith('robot_'): continue
        h = sha(f)
        rows.append(dict(name=m.get('name'), file=f, bytes=Path(f).stat().st_size, sha256=h, recon_lineage=recon.get(h), scan_only_match=scan_only.get(h),
                         also_copied_in_scan_dirs=scan_all.get(h) if h in recon else None))
    bad = [r for r in rows if r['scan_only_match'] or not r['recon_lineage']]
    txt = scene.read_text(); leaks = [k for k in ('gt_best_per_object', 'rackscan_20261021', 'build_rack_assets', 'swap_rack') if k in txt]
    return dict(scene=str(scene), recon_roots=[str(r) for r in RECON_ROOTS], forbidden_roots=[str(r) for r in FORBIDDEN_ROOTS],
                recon_mesh_hashes=len(recon), scan_mesh_hashes=len(scan_all), scan_only_mesh_hashes=len(scan_only), object_meshes=rows,
                violations=bad, scene_text_leaks=leaks, clean=not bad and not leaks)


def compare_with_prescan(w, epdir, scene):
    """Is this world the pre-scan world? tracking reference + object poses hashes, and the frozen scene text after path normalisation."""
    import re
    out = dict(prescan_episode_dir=str(PRESCAN_EPDIR))
    for f in ('tracking_reference.npz', 'reference_trajectory.npz', 'camera_trajectory.npz', 'metric_geometry.json'):
        a, b = epdir / f, PRESCAN_EPDIR / f
        out[f] = dict(identical_bytes=sha(a) == sha(b)) if a.exists() and b.exists() else None
    if not out['tracking_reference.npz']['identical_bytes']:
        import mujoco
        from tool_library.common import tracking as TR
        m = mujoco.MjModel.from_xml_path(str(scene))
        qa = TR.load_reference(epdir / 'tracking_reference.npz', m)['qpos']; qb = TR.load_reference(PRESCAN_EPDIR / 'tracking_reference.npz', m)['qpos']
        out['tracking_reference.npz']['max_abs_qpos_diff'] = float(np.abs(qa - qb).max()) if qa.shape == qb.shape else f'shape {qa.shape} vs {qb.shape}'
    teacher = R7 / 'diagnostics/cup_transfer/release_neutral/trial2/execution/scene_physics.xml'
    if teacher.exists():
        norm = lambda t: re.sub(r'file="[^"]*/(assets/[^"]*|[^"/]*)"', r'file="\1"', t)
        ta, tb = norm(scene.read_text()), norm(teacher.read_text())
        out['prescan_teacher_scene'] = dict(path=str(teacher), identical_after_path_normalisation=ta == tb)
        if ta != tb:
            import difflib
            d = [l for l in difflib.unified_diff(tb.splitlines(), ta.splitlines(), lineterm='', n=0) if not l.startswith(('---', '+++', '@@'))]
            out['prescan_teacher_scene']['diff_lines'] = len(d); out['prescan_teacher_scene']['diff_head'] = d[:30]
    return out


def frame0_and_settle(w, epdir, scene):
    import mujoco
    from scipy.spatial.transform import Rotation as R
    from tool_library.common import tracking as TR
    from floating_runtime import execute_reference
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    def bodyname(g): return m.body(int(m.geom_bodyid[g])).name
    def contacts(d):
        pens = []
        for c in d.contact[:d.ncon]:
            b1, b2 = bodyname(c.geom1), bodyname(c.geom2)
            if any(k in (b1, b2) for k in (CUP, OBJ)):
                pens.append(dict(geom1=m.geom(c.geom1).name, geom2=m.geom(c.geom2).name, body1=b1, body2=b2, dist_mm=float(c.dist * 1000)))
        pens.sort(key=lambda r: r['dist_mm']); summ = {}
        for r in pens: summ.setdefault('|'.join(sorted({r['body1'], r['body2']})), []).append(r['dist_mm'])
        return {k: dict(contacts=len(v), deepest_mm=min(v), count_below_minus1mm=int(sum(x < -1 for x in v))) for k, v in summ.items()}, pens[:40]
    f0, f0d = contacts(d)
    # set-down pose: objects at their LAST reference pose (cup inside the rack), robot at frame 0
    ref = TR.load_reference(epdir / 'tracking_reference.npz', m); q = ref['qpos']
    for o in (CUP, OBJ):
        adr = int(m.joint(o + '_free').qposadr[0]); d.qpos[adr:adr + 7] = q[-1, adr:adr + 7]
    mujoco.mj_forward(m, d); fl, fld = contacts(d)
    # settle A: robot frozen at frame 0 for 3 s, objects free from frame 0
    out = dict(frame0_contacts=f0, frame0_contacts_detail=f0d, final_pose_contacts=fl, final_pose_contacts_detail=fld)
    def settle(tag, qrow, n=61):
        qq = np.repeat(qrow[None], n, 0)
        sd = w / 'checks' / tag; sd.mkdir(parents=True)
        p = TR.save_reference(sd / 'reference.npz', m, qq, ref['labels'][:n], ref['fps'], ref['segment'][:n], ref['segments'], source=f'world check {tag}: robot frozen, objects free', floating_runtime=True, no_retiming=True)
        ex = execute_reference(str(scene), p, sd / 'execution')
        tr = np.load(ex['trajectory']); names = [str(x) for x in tr['object_names']]; tl = {}
        for i, nme in enumerate(names):
            p0 = tr['object_positions'][0, i]; r0 = R.from_quat(tr['object_quaternions_wxyz'][0, i][[1, 2, 3, 0]])
            tl[nme] = {f'{k/20:.1f}s': dict(cm=float(np.linalg.norm(tr['object_positions'][k, i] - p0) * 100),
                                        deg=float(np.degrees((R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]) * r0.inv()).magnitude())))
                       for k in range(0, n, 10)}
        return tl
    out['settle_drift_robot_frozen_frame0'] = settle('settle_frame0', q[0])
    # settle B: robot at frame 0 (hands far from the rack), cup + rack at their final reference pose: does our rack hold our cup where the reference says?
    qB = q[0].copy()
    for o in (CUP, OBJ):
        adr = int(m.joint(o + '_free').qposadr[0]); qB[adr:adr + 7] = q[-1, adr:adr + 7]
    out['settle_drift_objects_at_final_pose'] = settle('settle_final', qB)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rack-visual', type=Path, required=True); ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--allow-reuse-parts', action='store_true', help='if the mesh is byte-identical to the pre-scan recon rack, reuse its 128 validated CoACD parts')
    ap.add_argument('--threshold', type=float, default=.01); ap.add_argument('--max-hulls', type=int, default=128); ap.add_argument('--skip-checks', action='store_true')
    a = ap.parse_args(); w = a.workspace.resolve(); new_mesh = a.rack_visual.resolve()
    if w.exists(): raise SystemExit(f'workspace exists, never overwrite: {w}')
    for root in FORBIDDEN_ROOTS:
        if str(new_mesh).startswith(str(root)): raise SystemExit(f'refusing a mesh from a scan root: {new_mesh}')
    w.mkdir(parents=True); lineage = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), args=vars(a), started=time.strftime('%F %T'),
                                          hand_source=str(HAWOR), hand_source_hand_report_sha256=sha(HAWOR / 'hand_report.json'), cup_visual_sha256=sha(HAWOR / 'assets/blue_cup_visual.obj'))
    log('mesh frame report'); lineage['rack_mesh'] = mesh_frame_report(new_mesh); dump(w / 'lineage.json', lineage)
    if lineage['rack_mesh'].get('frame_warning'): log('WARNING: the new rack mesh is offset from the reference body frame', lineage['rack_mesh'].get('icp_new_to_reference'))
    log('stage source'); src = stage_hawor_source(w, new_mesh)
    log('coacd'); recon, reports, ident = coacd_parts(w, a.allow_reuse_parts, a.threshold, a.max_hulls, new_mesh)
    lineage['collision'] = {o: dict(generation=r['generation'], validation_passed=r['validation']['passed'],
                                    validation=str(w / 'collision_library' / NAME / o / 'validation.json'), **ident[o]) for o, r in reports.items()}
    lineage['collision']['merged'] = str(recon); dump(w / 'lineage.json', lineage)
    log('prepare episode dir'); res = prepare(w, w / 'collision_recon'); epdir = Path(res['input_dir']); lineage['prepare'] = res
    log('physics scene'); scene = build_scene(w, epdir); lineage['scene'] = str(scene)
    log('mesh manifest'); mm = mesh_manifest(w, epdir, scene); dump(w / 'world_meshes.json', mm); lineage['meshes_clean'] = mm['clean']
    if not mm['clean']: raise SystemExit('scan-derived mesh detected: ' + json.dumps(mm['violations'] + mm['scene_text_leaks']))
    log('compare with the pre-scan world'); lineage['prescan_comparison'] = compare_with_prescan(w, epdir, scene); dump(w / 'lineage.json', lineage)
    if not a.skip_checks:
        log('frame-0 / final-pose / settle checks'); chk = frame0_and_settle(w, epdir, scene); dump(w / 'world_check.json', chk)
        lineage['checks'] = {k: chk[k] for k in ('frame0_contacts', 'final_pose_contacts', 'settle_drift_robot_frozen_frame0', 'settle_drift_objects_at_final_pose')}
    lineage['finished'] = time.strftime('%F %T'); lineage['episode_dir'] = str(epdir); dump(w / 'lineage.json', lineage)
    log('DONE', json.dumps(dict(episode_dir=str(epdir), scene=str(scene), parts={o: v['parts'] for o, v in lineage['collision'].items() if isinstance(v, dict)}, clean=mm['clean'],
                                prescan=lineage['prescan_comparison'])))


if __name__ == '__main__': main()

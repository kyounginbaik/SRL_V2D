#!/usr/bin/env python3
# Adapted for episode 15 from streamN_ep16/code/build_world.py (only EP/NAME/source-tag changed; ep15 world_check adds nothing new).
"""Stream S (episode 15): build a complete physics world for episode 15 around a plastic_dish_rack VISUAL mesh of our own
reconstruction, with explicit lineage, and check it. Generalisation of streamI_ep30/code/build_world.py (episode 30) to
episode 15: no sweeper-track correction (episode 15 has none), the second object is the wooden_spoon, and the HaWoR source is
the stereo-v2 hand reconstruction that the pre-scan world physics_collision_v2/episodes/episode_000015 was built from
(collision_recon/episode_000015/collision_repair_provenance.json: source_hand_reconstruction = hawor_stereo_recon_v2/episode_000015,
hand_report sha256 cebeffcf...). Nothing scan-derived is read: hands, object poses, camera and the wooden_spoon assets come from
that source (the user's final_recon_ours ZIP objects); the rack visual mesh comes from --rack-visual; its collision parts are
CoACD-decomposed from that mesh by the unchanged repair_collision_assets.py (threshold 0.01, 128 hulls, seed 44 like the shipped
recon world's collision_library) or, with --allow-reuse-parts and a byte-identical mesh, the validated 128 parts of
collision_library/episode_000015 are reused; the episode dir is prepared by the unchanged prepare_seven.py; the physics scene is the
unchanged floating_runtime._freeze + calibrate_controller.controller_scene('kp5000_r200_d5') (the run_seven_mpc baseline recipe).

Body frame convention (MUST hold for a new mesh): the rack mesh vertices are expressed in the rack body frame of
object_poses.npz (world_from_object of 'plastic_dish_rack'), metric metres, no scale, no visual origin offset. Reference object
poses are never changed. The current reconstruction mesh (sha256 3790d7e0... (ep15 recon rack)) defines that frame; a new mesh is compared against
it (bounds, centroid, rigid ICP offset) and the world is checked at frame 0 (penetrations, 3 s settle with the robot held still)
and at the final reference pose of the spoon (rest probe: is the placed spoon supported by the rack?).

usage: build_world.py --rack-visual MESH.obj --workspace OUT_DIR [--allow-reuse-parts] [--threshold 0.01] [--max-hulls 128] [--skip-checks]
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, hashlib, json, shutil, sys, time, types
from pathlib import Path
import numpy as np

R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
EP = 15; NAME = 'episode_000015'; OBJ = 'plastic_dish_rack'; OTHER = 'wooden_spoon'
HAWOR = R7 / 'hawor_stereo_recon_v2' / NAME                     # stereo-v2 HaWoR hands + ZIP objects (no scan anywhere)
REF_RACK = R7 / 'collision_recon' / NAME / 'assets/plastic_dish_rack_visual.obj'   # 3790d7e0... (ep15 recon rack), defines the body frame
REF_PARTS = R7 / 'collision_library' / NAME / OBJ / 'parts'                         # validated CoACD parts of REF_RACK (128)
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
    """Copy of the HaWoR/ZIP source with the rack visual replaced (bytes verbatim) and the coarse ZIP collision replaced by one
    convex hull of the new mesh (only used as the 'old' baseline inside the CoACD validation; not shipped)."""
    import trimesh
    dst = w / 'source_hawor' / NAME
    shutil.copytree(HAWOR, dst, ignore=shutil.ignore_patterns('archive_wilor', 'archive_before_stereo', '*.mp4', '__pycache__', '*.pyc'))
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
    (dst / 'STREAM_S_SOURCE.txt').write_text(f'Copy of {HAWOR} with assets/{OBJ}_visual.obj and {OBJ}.obj replaced by {new_mesh} '
        f'(sha256 {sha(new_mesh)}); hands, camera, object poses, reference.parquet and {OTHER} assets byte-identical to the source.\n')
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
    # also: are the rack parts byte-identical to the shipped pre-scan library?
    lib = {sha(p): p.name for p in REF_PARTS.glob('part_*.obj')}
    same = sum(1 for r in rows if r['sha256'] in lib)
    return dict(scene=str(scene), forbidden_roots=[str(r) for r in FORBIDDEN_ROOTS], forbidden_rack_hashes_checked=len(forb), object_meshes=rows,
                scan_matches=bad, clean=not bad, parts_identical_to_prescan_library=same, prescan_library=str(REF_PARTS))


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
        if any(k in (b1, b2) for k in (OTHER, OBJ)):
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
        p0 = tr['object_positions'][0, i]; q0 = tr['object_quaternions_wxyz'][0, i]; r0 = R.from_quat(q0[[1, 2, 3, 0]])
        tl[n] = {f'{k/20:.1f}s': dict(cm=float(np.linalg.norm(tr['object_positions'][k, i] - p0) * 100),
                                  deg=float(np.degrees((R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]) * r0.inv()).magnitude())))
                 for k in range(0, 61, 10)}
    # rest probe: spoon placed at its reference pose of frame K (final placement in the rack), hands parked far away, 2 s settle
    oc = int(m.joint(f'{OTHER}_free').qposadr[0]); rc = int(m.joint(f'{OBJ}_free').qposadr[0]); qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    refq = ref['qpos']; rest = {}
    for k in (0, len(refq) - 1):
        dd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dd, 0); dd.qpos[:] = refq[0]; dd.qpos[oc:oc + 7] = refq[k, oc:oc + 7]
        for s in ('left', 'right'):
            for ax, v in (('x', 0.6 * (1 if s == 'right' else -1)), ('y', -0.6), ('z', 0.5)): dd.qpos[int(m.joint(f'robot_{s}_{s}_pos_{ax}').qposadr[0])] = v
        dd.qvel[:] = 0; mujoco.mj_forward(m, dd); dd.ctrl[:] = dd.qpos[qa]
        pen0 = min([c.dist for c in dd.contact[:dd.ncon] if OTHER in (bodyname(c.geom1), bodyname(c.geom2))], default=0.) * 1000
        for _ in range(int(2.0 / m.opt.timestep)): mujoco.mj_step(m, dd)
        pp = dd.qpos[oc:oc + 3]; qq = dd.qpos[oc + 3:oc + 7]
        e = float(np.linalg.norm(pp - refq[k, oc:oc + 3]) * 100); er = float(np.degrees((R.from_quat(qq, scalar_first=True) * R.from_quat(refq[k, oc + 3:oc + 7], scalar_first=True).inv()).magnitude()))
        rr = float(np.linalg.norm(dd.qpos[rc:rc + 3] - refq[k, rc:rc + 3]) * 100)
        rest[f'frame_{k}'] = dict(t_s=k / 20., spoon_deepest_penetration_mm=float(pen0), settled_err_cm=e, settled_err_deg=er, dz_cm=float((pp[2] - refq[k, oc + 2]) * 100), rack_moved_cm=rr)
    return dict(frame0_contacts=summary, frame0_contacts_detail=pens[:40], settle_drift_robot_frozen=tl, settle_execution=ex['scene'], rest_probe_hands_parked=rest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rack-visual', type=Path, required=True); ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--allow-reuse-parts', action='store_true', help='if the mesh is byte-identical to the shipped recon rack, reuse its validated CoACD parts')
    ap.add_argument('--threshold', type=float, default=.01); ap.add_argument('--max-hulls', type=int, default=128); ap.add_argument('--skip-checks', action='store_true')
    a = ap.parse_args(); w = a.workspace.resolve(); new_mesh = a.rack_visual.resolve()
    if w.exists(): raise SystemExit(f'workspace exists, never overwrite: {w}')
    w.mkdir(parents=True); lineage = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), args=vars(a), started=time.strftime('%F %T'),
                                          hawor_source=str(HAWOR), hawor_hand_report_sha256=sha(HAWOR / 'hand_report.json'), hawor_reference_parquet_sha256=sha(HAWOR / 'reference.parquet'),
                                          hawor_object_poses_sha256=sha(HAWOR / 'object_poses.npz'))
    log('mesh frame report'); lineage['rack_mesh'] = mesh_frame_report(new_mesh); dump(w / 'lineage.json', lineage)
    if lineage['rack_mesh'].get('frame_warning'): log('WARNING: the new rack mesh is offset from the reference body frame', lineage['rack_mesh'].get('icp_new_to_reference'))
    log('stage source'); src = stage_hawor_source(w, new_mesh)
    log('coacd'); recon, report = coacd_parts(w, a.allow_reuse_parts, a.threshold, a.max_hulls, new_mesh)
    lineage['collision'] = dict(parts=len(report['convex_parts']), generation=report['generation'], validation_passed=report['validation']['passed'],
                                validation=str(w / 'collision_library' / NAME / OBJ / 'validation.json'), merged=str(recon))
    dump(w / 'lineage.json', lineage)
    log('prepare episode dir'); res = prepare(w, w / 'collision_recon'); epdir = Path(res['input_dir']); lineage['prepare'] = res
    log('physics scene'); scene = build_scene(w, epdir); lineage['scene'] = str(scene)
    log('mesh manifest'); mm = mesh_manifest(w, epdir, scene); dump(w / 'world_meshes.json', mm); lineage['meshes_clean'] = mm['clean']
    if not mm['clean']: raise SystemExit('scan-derived mesh detected: ' + json.dumps(mm['scan_matches']))
    if not a.skip_checks:
        log('frame-0 + settle + rest checks'); chk = frame0_and_settle(w, epdir, scene); dump(w / 'world_check.json', chk)
        lineage['checks'] = dict(frame0=chk['frame0_contacts'], settle=chk['settle_drift_robot_frozen'], rest=chk['rest_probe_hands_parked'])
    lineage['finished'] = time.strftime('%F %T'); lineage['episode_dir'] = str(epdir); dump(w / 'lineage.json', lineage)
    log('DONE', json.dumps(dict(episode_dir=str(epdir), scene=str(scene), parts=lineage['collision']['parts'], clean=mm['clean'], parts_identical=mm['parts_identical_to_prescan_library'])))


if __name__ == '__main__': main()

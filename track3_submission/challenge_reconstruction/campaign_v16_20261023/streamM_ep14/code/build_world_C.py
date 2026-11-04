#!/usr/bin/env python3
"""Stream M phase 6: build the episode-14 world around a CORRECTED source dir (compose-retargeted object reference made by
compose_reference.py from world_6prong/collision_recon). The source already holds the merged, validated collision assets of
world_6prong (six-prong rack 090b7d17 + its 128 CoACD parts, cup 78bd065c + 42 pre-scan parts), so no decomposition is run and the
geometry is byte-identical to world_6prong; only the object reference (cup track, rack placement) differs.
Uses build_world.py's prepare / build_scene / mesh_manifest / frame0_and_settle unchanged.
usage: build_world_C.py SOURCE_ROOT(dir containing episode_000014) WORKSPACE"""
import sys, json, time, importlib.util
from pathlib import Path
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bw', here / 'build_world.py'); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
src_root, w = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if w.exists(): raise SystemExit('exists ' + str(w))
w.mkdir(parents=True); ep = src_root / B.NAME
corr = json.loads((ep / 'compose_target_correction.json').read_text())
lineage = dict(script=str(Path(__file__).resolve()), script_sha256=B.sha(__file__), build_world_sha256=B.sha(here / 'build_world.py'), source=str(ep),
               correction=dict(path=str(ep / 'compose_target_correction.json'), sha256=B.sha(ep / 'compose_target_correction.json'),
                               corrected_object_poses_sha256=corr['corrected_object_poses_sha256'], original_object_poses_sha256=corr['original_object_poses_sha256']),
               rack_visual_sha256=B.sha(ep / 'assets/plastic_dish_rack_visual.obj'), cup_visual_sha256=B.sha(ep / 'assets/blue_cup_visual.obj'), started=time.strftime('%F %T'))
B.log('prepare'); res = B.prepare(w, src_root); epdir = Path(res['input_dir']); lineage['prepare'] = res
B.log('scene'); scene = B.build_scene(w, epdir); lineage['scene'] = str(scene)
W6 = here.parent / 'world_6prong'
extra = [Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/rack6prong_meshes'), W6 / 'collision_library' / B.NAME / B.OBJ, ep / 'assets']
B.log('mesh manifest'); mm = B.mesh_manifest(w, epdir, scene, extra)
# the corrected source's own assets are in 'extra' only to resolve paths; require every mesh to ALSO exist in world_6prong (byte-identical geometry)
w6 = {}
for p in (W6 / 'collision_recon' / B.NAME / 'assets').rglob('*'):
    if p.is_file() and p.suffix == '.obj': w6.setdefault(B.sha(p), str(p))
for r in mm['object_meshes']: r['world_6prong_identical'] = w6.get(r['sha256'])
mm['all_identical_to_world_6prong'] = all(r['world_6prong_identical'] for r in mm['object_meshes'])
B.dump(w / 'world_meshes.json', mm)
if not mm['clean'] or not mm['all_identical_to_world_6prong']: raise SystemExit('mesh check failed')
B.log('checks'); chk = B.frame0_and_settle(w, epdir, scene); B.dump(w / 'world_check.json', chk)
lineage['checks'] = {k: chk[k] for k in ('frame0_contacts', 'final_pose_contacts', 'settle_drift_robot_frozen_frame0')}
lineage['finished'] = time.strftime('%F %T'); lineage['episode_dir'] = str(epdir); B.dump(w / 'lineage.json', lineage)
B.log('DONE', json.dumps(dict(clean=mm['clean'], identical=mm['all_identical_to_world_6prong'], frame0=chk['frame0_contacts'], final=chk['final_pose_contacts'])))

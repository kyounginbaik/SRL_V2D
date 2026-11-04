#!/usr/bin/env python3
"""Stream Q phase 6 (v21C hedge, ep25): world whose reference = the coworker's compose_eval board tracks, with OUR slotted boards.
(1) reference: compose_to_world.build() (compose T_world_obj -> our slotted-mesh body frames by the near-identity ICP registration,
    -> our world by one rigid transform = frame-0 camera chain + table correction resting the compose boards on z=0);
(2) corrected source copy of collision_recon_slots_v3/episode_000025: object_poses / trajectories / reference.parquet object columns
    replaced, assets wooden_piece_X.obj + _visual.obj replaced by the slotted meshes (meshes_slotted_C, same body frames),
    board_track_correction.json (lineage: compose shas, G, X, table correction, per-frame shift vs the v21 reference, mesh shas);
(3) unchanged prepare_seven -> episodes/episode_000025;
(4) scene = the v19/w3 frozen scene with: board-1 collision parts -> collision_C/wooden_piece_1_cutpz (slot +z at x=0, same part count),
    board-2 collision unchanged (cut3mm, slot +z), visual meshes -> the new episode's slotted stage2 assets, keyframe object values ->
    the new frame 0; (5) checks: keyframe == reference frame 0, frame-0 contacts, 2 s settle with the robot frozen.
usage: build_worldC.py --workspace OUT"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, hashlib, json, shutil, sys, re, time
from pathlib import Path
import numpy as np, trimesh, pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; Q = HERE.parent
R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
SRC = R7 / 'collision_recon_slots_v3/episode_000025'; V21 = Q / 'world/w4/source_corrected/episode_000025/object_poses.npz'
OLD_SCENE = Q / 'world/w3/scene/scene_physics_corrected.xml'; CT = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000025')
sys.path[:0] = [str(HERE), str(R7 / 'code'), str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
NAMES = ['wooden_piece_1', 'wooden_piece_2']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--workspace', type=Path, required=True); a = ap.parse_args(); w = a.workspace.resolve()
    if w.exists(): raise SystemExit('exists: ' + str(w))
    w.mkdir(parents=True); lin = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), started=time.strftime('%F %T'))
    import compose_to_world as CW
    T, G, info = CW.build()
    dst = w / 'source_corrected/episode_000025'; shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('__pycache__'))
    z = dict(np.load(SRC / 'object_poses.npz')); T0 = z['world_from_object']; z['world_from_object'] = T; np.savez_compressed(dst / 'object_poses.npz', **z)
    for i, n in enumerate(NAMES):
        tr = dict(np.load(SRC / f'{n}_trajectory.npz')); tr['T_world_obj'] = T[:, i].astype(tr['T_world_obj'].dtype)
        if 'T_rgb_obj' in tr: tr['T_rgb_obj'] = (np.linalg.inv(z['world_from_rgb'].astype(float)) @ T[:, i]).astype(tr['T_rgb_obj'].dtype)
        np.savez_compressed(dst / f'{n}_trajectory.npz', **tr)
        for f in (f'{n}.obj', f'{n}_visual.obj'): shutil.copyfile(Q / f'meshes_slotted_C/{n}_visual_slotted.obj', dst / 'assets' / f)
    tab = pq.read_table(SRC / 'reference.parquet'); row = tab.to_pylist()[0]; bn = list(row['object_body_names']); pos = np.array(row['object_body_position']); q = np.array(row['object_body_wxyz'])
    for i, n in enumerate(NAMES):
        j = bn.index(n); pos[:, j] = T[:, i, :3, 3]; q[:, j] = R.from_matrix(T[:, i, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        if j == 0 and bn[0] == row['object_name']: row['object_root_position'] = pos[:, 0].tolist(); row['object_root_axis_angle'] = R.from_matrix(T[:, i, :3, :3]).as_rotvec().tolist()
    row['object_body_position'] = pos.tolist(); row['object_body_wxyz'] = q.tolist(); pq.write_table(pa.Table.from_pylist([row], schema=tab.schema), dst / 'reference.parquet')
    V = np.load(V21)['world_from_object']; shift = {}
    for i, n in enumerate(NAMES):
        dp = np.linalg.norm(T[:, i, :3, 3] - V[:, i, :3, 3], axis=1) * 100; dr = np.degrees((R.from_matrix(T[:, i, :3, :3]) * R.from_matrix(V[:, i, :3, :3]).inv()).magnitude())
        shift[n] = dict(mean_cm=float(dp.mean()), max_cm=float(dp.max()), mean_deg=float(dr.mean()), max_deg=float(dr.max()), per_second=[[s, round(float(dp[s*20:(s+1)*20].mean()), 2), round(float(dr[s*20:(s+1)*20].mean()), 1)] for s in range(18)])
    rec = dict(schema='ep25_compose_target_reference_v1', episode=25, purpose='v21C hedge: reference = coworker compose_eval board tracks in our world/mesh frames',
               script=lin['script'], script_sha256=lin['script_sha256'], compose_to_world=str(HERE / 'compose_to_world.py'), compose_to_world_sha256=sha(HERE / 'compose_to_world.py'),
               compose_tracks={n: dict(path=str(CT / f'{n}_trajectory.npz'), sha256=sha(CT / f'{n}_trajectory.npz')) for n in NAMES}, compose_camera_sha256=sha(CT / 'camera_motion.npz'),
               compose_meshes_used_only_for_registration={n: sha(CT / f'{n}_mesh_metric.obj') for n in NAMES},
               mesh_registration_G_ours_to_compose={n: G[n].tolist() for n in NAMES}, transform_info=info,
               submitted_meshes={n: dict(path=str(Q / f'meshes_slotted_C/{n}_visual_slotted.obj'), sha256=sha(Q / f'meshes_slotted_C/{n}_visual_slotted.obj')) for n in NAMES},
               slotted_mesh_report=json.loads((Q / 'meshes_slotted_C/slotted_mesh_report.json').read_text()),
               shift_vs_v21_reference_w4=shift, original_object_poses_sha256=sha(SRC / 'object_poses.npz'), corrected_object_poses_sha256=sha(dst / 'object_poses.npz'),
               original_reference_parquet_sha256=sha(SRC / 'reference.parquet'), corrected_reference_parquet_sha256=sha(dst / 'reference.parquet'))
    dump(dst / 'board_track_correction.json', rec); lin['reference'] = {k: rec[k] for k in ('corrected_object_poses_sha256', 'shift_vs_v21_reference_w4', 'submitted_meshes')}
    log('shift vs v21', json.dumps({n: {k: v for k, v in s.items() if k != 'per_second'} for n, s in shift.items()}))
    import prepare_seven as PS
    PS.BASE = w; res = PS.prepare_episode(25, w / 'source_corrected'); epdir = Path(res['input_dir']); shutil.copyfile(R7 / 'input_manifest.json', w / 'input_manifest.json'); lin['prepare'] = res
    import mujoco
    from tool_library.common import tracking as TR
    txt = OLD_SCENE.read_text()
    txt = txt.replace(str(R7 / 'ep25_v15/collision/wooden_piece_1_cut3mm'), str(Q / 'collision_C/wooden_piece_1_cutpz'))
    for n in NAMES:
        txt = txt.replace(str(R7 / f'physics_slots_v3/episodes/episode_000025/stage2/assets/{n}/{n}_visual.obj'), str(epdir / f'stage2/assets/{n}/{n}_visual.obj'))
    m = mujoco.MjModel.from_xml_path(str(OLD_SCENE)); new = TR.load_reference(epdir / 'tracking_reference.npz', m)['qpos']
    mk = re.search(r'(<key name="home" qpos=")([^"]*)(")', txt); vals = mk.group(2).split()
    for n in NAMES:
        adr = int(m.joint(n + '_free').qposadr[0])
        for j in range(7): vals[adr + j] = repr(float(new[0, adr + j]))
    txt = txt[:mk.start(2)] + ' '.join(vals) + txt[mk.end(2):]
    sd = w / 'scene'; sd.mkdir(); scene = sd / 'scene_physics_C.xml'; scene.write_text(txt)
    assert (epdir / 'stage2/assets/wooden_piece_1/wooden_piece_1_visual.obj').exists()
    m2 = mujoco.MjModel.from_xml_path(str(scene)); d2 = mujoco.MjData(m2); mujoco.mj_resetDataKeyframe(m2, d2, 0); mujoco.mj_forward(m2, d2)
    pen = {}
    for c in d2.contact[:d2.ncon]:
        b1, b2 = m2.body(int(m2.body_rootid[m2.geom_bodyid[c.geom1]])).name, m2.body(int(m2.body_rootid[m2.geom_bodyid[c.geom2]])).name; key = '|'.join(sorted({b1, b2})); pen[key] = min(pen.get(key, 0.), float(c.dist * 1000))
    lin['scene'] = dict(path=str(scene), sha256=sha(scene), from_scene=str(OLD_SCENE), keyframe_max_abs_vs_reference=float(np.abs(d2.qpos - new[0]).max()), frame0_deepest_contacts_mm=pen,
                        visual_meshes={n: sha(epdir / f'stage2/assets/{n}/{n}_visual.obj') for n in NAMES})
    log('scene', json.dumps(lin['scene']))
    from build_world_q import settle
    lin['settle_robot_frozen'] = settle(scene, epdir, w / 'checks/settle')
    lin['episode_dir'] = str(epdir); lin['finished'] = time.strftime('%F %T'); dump(w / 'lineage.json', lin); log('DONE', json.dumps(lin['settle_robot_frozen']))
if __name__ == '__main__': main()

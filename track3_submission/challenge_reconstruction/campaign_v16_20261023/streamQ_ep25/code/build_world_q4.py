#!/usr/bin/env python3
"""Stream Q ep25 (v4 corrector correct_ep25_v4.py): corrected-reference world. (1) correct_ep25.py on the slots_v3 source (user ZIP poses) -> WORKSPACE/source_corrected;
(2) unchanged prepare_seven.prepare_episode with BASE=WORKSPACE -> WORKSPACE/episodes/episode_000025 (corrected tracking reference,
input_manifest new_input_provenance.hawor_source = the corrected source, which holds board_track_correction.json);
(3) physics scene = the v19 teacher's frozen scene (slot-cut collision, kp5000 controller) with ONLY the two object free-joint values of
the keyframe replaced by the corrected frame-0 poses (corrected frame 0 = boards resting flat; equals the old scene's physically settled
state within 1 mm / 0.13 deg); (4) checks: keyframe == corrected reference frame 0, 2 s settle with the robot frozen (old vs new scene).
usage: build_world_q.py --workspace OUT [--evidence JSON ...]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, hashlib, json, subprocess, sys, re, time
from pathlib import Path
import numpy as np
R7 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
SRC = R7 / 'collision_recon_slots_v3/episode_000025'
OLD_EP = R7 / 'physics_slots_v3/episodes/episode_000025'
OLD_SCENE = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamF_ep25/search/hLR_R_13.0_17.75_s3/execution/scene_physics.xml')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R7 / 'code'), str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

def settle(scene, epdir, out, seconds=2.0):
    import mujoco
    from tool_library.common import tracking as TR
    from floating_runtime import execute_reference
    from scipy.spatial.transform import Rotation as R
    m = mujoco.MjModel.from_xml_path(str(scene)); ref = TR.load_reference(Path(epdir) / 'tracking_reference.npz', m)
    n = int(seconds * 20) + 1; q = ref['qpos'][:n].copy(); q[:] = q[0]
    out.mkdir(parents=True)
    p = TR.save_reference(out / 'reference.npz', m, q, ref['labels'][:n], ref['fps'], ref['segment'][:n], ref['segments'], source='robot frozen at frame 0, objects free', floating_runtime=True, no_retiming=True)
    ex = execute_reference(str(scene), p, out / 'execution'); tr = np.load(ex['trajectory']); res = {}
    for i, nm in enumerate([str(x) for x in tr['object_names']]):
        p0 = tr['object_positions'][0, i]; r0 = R.from_quat(tr['object_quaternions_wxyz'][0, i][[1, 2, 3, 0]])
        res[nm] = {f'{k/20:.2f}s': [round(float(np.linalg.norm(tr['object_positions'][k, i] - p0) * 100), 3),
                                    round(float(np.degrees((R.from_quat(tr['object_quaternions_wxyz'][k, i][[1, 2, 3, 0]]) * r0.inv()).magnitude())), 3)] for k in (1, 2, 5, 10, 20, n - 1)}
    return res

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--workspace', type=Path, required=True); ap.add_argument('--evidence', type=Path, nargs='*', default=[])
    a = ap.parse_args(); w = a.workspace.resolve()
    if w.exists(): raise SystemExit('workspace exists, never overwrite: ' + str(w))
    w.mkdir(parents=True); lin = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__), started=time.strftime('%F %T'))
    log('correct'); subprocess.run([sys.executable, str(HERE / 'correct_ep25_v4.py'), '--source', str(SRC), '--output', str(w / 'source_corrected/episode_000025'), '--evidence', *map(str, a.evidence)], check=True)
    corr = json.loads((w / 'source_corrected/episode_000025/board_track_correction.json').read_text())
    lin['correction'] = dict(path=str(w / 'source_corrected/episode_000025/board_track_correction.json'), corrected_object_poses_sha256=corr['corrected_object_poses_sha256'], reference_shift=corr['reference_shift'])
    log('prepare'); import prepare_seven as PS
    PS.BASE = w; res = PS.prepare_episode(25, w / 'source_corrected'); epdir = Path(res['input_dir']); lin['prepare'] = res
    import shutil; shutil.copyfile(R7 / 'input_manifest.json', w / 'input_manifest.json')
    # compare with the old episode dir: only the object columns may differ
    import mujoco
    from tool_library.common import tracking as TR
    m = mujoco.MjModel.from_xml_path(str(OLD_SCENE))
    new = TR.load_reference(epdir / 'tracking_reference.npz', m)['qpos']; old = TR.load_reference(OLD_EP / 'tracking_reference.npz', m)['qpos']
    oc = {o: int(m.joint(o + '_free').qposadr[0]) for o in ('wooden_piece_1', 'wooden_piece_2')}; mask = np.ones(new.shape[1], bool)
    for adr in oc.values(): mask[adr:adr + 7] = False
    lin['tracking_reference_robot_columns_identical'] = bool(np.array_equal(new[:, mask], old[:, mask]))
    lin['tracking_reference_object_columns_max_change'] = float(np.abs(new[:, ~mask] - old[:, ~mask]).max())
    log('robot columns identical', lin['tracking_reference_robot_columns_identical'])
    # scene: keyframe object values only
    sd = w / 'scene'; sd.mkdir(); txt = OLD_SCENE.read_text()
    mk = re.search(r'(<key name="home" qpos=")([^"]*)(")', txt); vals = mk.group(2).split(); q0 = np.array(vals, float)
    assert len(q0) == m.nq and np.allclose(q0, old[0], atol=1e-5)
    for o, adr in oc.items():
        for j in range(7): vals[adr + j] = repr(float(new[0, adr + j]))
    txt2 = txt[:mk.start(2)] + ' '.join(vals) + txt[mk.end(2):]; scene = sd / 'scene_physics_corrected.xml'; scene.write_text(txt2)
    m2 = mujoco.MjModel.from_xml_path(str(scene)); d2 = mujoco.MjData(m2); mujoco.mj_resetDataKeyframe(m2, d2, 0)
    lin['scene'] = dict(path=str(scene), sha256=sha(scene), from_scene=str(OLD_SCENE), from_scene_sha256=sha(OLD_SCENE), only_keyframe_object_values_changed=True,
                        keyframe_equals_corrected_reference_frame0=bool(np.allclose(d2.qpos, new[0], atol=1e-9, rtol=0)),
                        lines_changed=int(sum(x != y for x, y in zip(txt.splitlines(), txt2.splitlines()))))
    log('scene', lin['scene'])
    log('settle checks'); lin['settle_robot_frozen'] = dict(old_scene_old_reference=settle(OLD_SCENE, OLD_EP, w / 'checks/settle_old'), new_scene_corrected=settle(scene, epdir, w / 'checks/settle_new'))
    lin['episode_dir'] = str(epdir); lin['finished'] = time.strftime('%F %T'); dump(w / 'lineage.json', lin); log('DONE', json.dumps(lin['settle_robot_frozen']))

if __name__ == '__main__': main()

"""Stream K phase 6 (v21C, ep35): re-target the spoon reference to the coworker's compose_eval track.
1. Mesh registration exactly as the scorer: our submitted mesh (object_mesh_b64 of the packed submission) registered onto the compose
   mesh with MP-SR._register_object_body (same seed 1_000_003*ep+101*slot, same compose geometry as score_vs_compose.py) -> (ca, rot).
   Compose pose (R_c, p_c) of their mesh <-> our body pose R_o = R_c rot, p_o = p_c + R_c anchor - R_o ca (scorer's body conversion).
2. One rigid transform per episode A = W_ours(0) T_o(0)^-1 (source world), so frame 0 equals our current reference (spoon start
   unchanged); the scorer's frame-0 anchor alignment makes the scored error invariant to this choice.
3. Report the per-frame shift vs the v21 reference, the end-rest height/tilt vs frame 0 (table consistency).
usage: build_target.py SUB_PARQUET OUT_JSON_NPZ_PREFIX [--write-source OUT_DIR]"""
import sys, json, hashlib, shutil
import numpy as np, pandas as pd, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
sys.argv_ = sys.argv
EP = 35; SUB = sys.argv[1]; OUT = Path(sys.argv[2])
CMP = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000035')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000035')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
M = K.load_metric('MP-SR')
sub = pd.read_parquet(SUB); s = sub[sub.row_id.str.contains('/e000035/')]
enc = s.object_mesh_b64.fillna('').astype(str); cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0])
m = trimesh.load(CMP / 'wooden_spoon_mesh_metric.obj', force='mesh'); m.merge_vertices()
if len(m.faces) > 40000:
    try: m = m.simplify_quadric_decimation(face_count=40000)
    except Exception: pass
m.remove_unreferenced_vertices(); rv = np.asarray(m.vertices, np.float64); rf = np.asarray(m.faces, np.int64)
anchor, _ = M._surface_moments(rv, rf)
ca, rot = M._register_object_body(cand, (rv, rf), anchor, 1_000_003 * EP + 101 * 0)
Tc = np.load(CMP / 'wooden_spoon_trajectory.npz')['T_world_obj']; n = len(Tc)
Ro = Tc[:, :3, :3] @ rot; po = Tc[:, :3, 3] + Tc[:, :3, :3] @ anchor - Ro @ ca
To = np.tile(np.eye(4), (n, 1, 1)); To[:, :3, :3] = Ro; To[:, :3, 3] = po
z = dict(np.load(SRC / 'object_poses.npz', allow_pickle=False)); W = z['world_from_object'][:, 0]; assert len(W) == n, (len(W), n)
A = W[0] @ np.linalg.inv(To[0]); Wn = A @ To
dp = np.linalg.norm(Wn[:, :3, 3] - W[:, :3, 3], axis=1) * 100
dr = np.degrees(R.from_matrix(np.einsum('nji,njk->nik', W[:, :3, :3], Wn[:, :3, :3])).magnitude())
# table consistency in the physics (setup) frame: lowest mesh point height and up-tilt of the spoon at rest frames
man = json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter/input_manifest.json').read_text())
S = np.asarray(man['setup_from_source']); ours = trimesh.load(SRC / 'assets/wooden_spoon.obj', force='mesh'); V = np.asarray(ours.vertices)
def low(T): X = (S @ T)[:3, :3] @ V.T + (S @ T)[:3, 3:4]; return X[2].min() * 100
rep = dict(compose_track=str(CMP / 'wooden_spoon_trajectory.npz'), compose_track_sha256=sha(CMP / 'wooden_spoon_trajectory.npz'),
           compose_mesh_sha256=sha(CMP / 'wooden_spoon_mesh_metric.obj'), source_object_poses_sha256=sha(SRC / 'object_poses.npz'),
           registration=dict(candidate_anchor=ca, body_rotation=rot, body_rotation_deg=float(np.degrees(R.from_matrix(rot).magnitude())), compose_anchor=anchor, seed=1_000_003 * EP),
           world_transform_source_from_compose=A, shift_vs_v21_cm=dict(mean=float(dp.mean()), median=float(np.median(dp)), max=float(dp.max())),
           shift_vs_v21_deg=dict(mean=float(dr.mean()), median=float(np.median(dr)), max=float(dr.max())),
           lowest_point_setup_cm=dict(v21=[round(low(W[k]), 2) for k in (0, 20, 100, 200, 300, 380, 400, 420)], new=[round(low(Wn[k]), 2) for k in (0, 20, 100, 200, 300, 380, 400, 420)]))
print(json.dumps(rep, indent=1, default=lambda x: np.asarray(x).round(5).tolist()))
for s0 in range(0, n, 20): print(f'{s0/20:5.1f}s shift {dp[s0:s0+20].mean():5.2f} cm {dr[s0:s0+20].mean():5.1f} deg')
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(str(OUT) + '.npz', world_from_object_new=Wn, world_from_object_v21=W, compose_T=Tc, A=A, ca=ca, rot=rot)
Path(str(OUT) + '.json').write_text(json.dumps(rep, indent=1, default=lambda x: np.asarray(x).tolist()))
if '--write-source' in sys.argv:
    D = Path(sys.argv[sys.argv.index('--write-source') + 1]); assert not D.exists(), D
    shutil.copytree(SRC, D, symlinks=True); z['world_from_object'] = z['world_from_object'].copy(); z['world_from_object'][:, 0] = Wn
    np.savez_compressed(D / 'object_poses.npz', **z)
    rec = dict(rep, episode=35, object='wooden_spoon', method='compose_eval_20261021 spoon track converted into our mesh body frame by the scorer registration and into our source world by one rigid transform fixing frame 0',
               corrected_object_poses_sha256=sha(D / 'object_poses.npz'), script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
    (D / 'compose_retarget_lineage.json').write_text(json.dumps(rec, indent=1, default=lambda x: np.asarray(x).tolist())); print('wrote', D)

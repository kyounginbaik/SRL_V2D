"""Build the ep35 adapter episode dir for the compose-retargeted source: copy of refinement_20261018/wrist_chart/episode_000035/
chart_0/adapter with the spoon free-joint columns of tracking_reference.npz replaced by setup_from_source @ world_from_object (new),
manifest source / source_object_poses_sha256 pointed at the corrected source copy. Hand reference, scene, setup unchanged.
usage: build_episode.py SOURCE_DIR OUT_EPISODE_DIR"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
SRC, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); assert not OUT.exists(), OUT
A = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copytree(A, OUT)
man = json.loads((A / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source'])
W = np.load(SRC / 'object_poses.npz')['world_from_object'][:, 0]; P = np.einsum('ij,njk->nik', S, W)
z = dict(np.load(A / 'tracking_reference.npz', allow_pickle=False)); Q = z['qpos'].copy()
jn = [str(x) for x in z['joint_names']]; c = jn.index('wooden_spoon_free[0]')
q = np.roll(R.from_matrix(P[:, :3, :3]).as_quat(), 1, axis=1)
for k in range(1, len(q)):
    if np.dot(q[k], q[k - 1]) < 0: q[k] = -q[k]
if np.dot(q[0], Q[0, c + 3:c + 7]) < 0: q = -q
Q[:, c:c + 3] = P[:, :3, 3]; Q[:, c + 3:c + 7] = q
z['qpos'] = Q; z['source'] = np.array('Original hand targets in a nonsingular virtual wrist chart; spoon targets = compose_eval_20261021 track re-targeted (streamK_ep35/C/code/build_target.py)')
np.savez_compressed(OUT / 'tracking_reference.npz', **z)
man['source'] = str(SRC); man['source_object_poses_sha256'] = sha(SRC / 'object_poses.npz')
man['compose_retarget'] = dict(lineage=str(SRC / 'compose_retarget_lineage.json'), lineage_sha256=sha(SRC / 'compose_retarget_lineage.json'), original_adapter=str(A))
(OUT / 'input_manifest.json').write_text(json.dumps(man, indent=1))
print('wrote', OUT, 'max |dq| spoon pos vs original (cm)', float(np.abs(Q[:, c:c + 3] - np.load(A / 'tracking_reference.npz')['qpos'][:, c:c + 3]).max() * 100))

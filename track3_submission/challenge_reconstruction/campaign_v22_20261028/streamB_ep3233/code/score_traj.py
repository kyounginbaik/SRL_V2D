"""usage: score_traj.py EP DIR... : scorer-faithful (MP-SR registration + spoon-anchored frame-0 alignment) mean errors vs target_final_v5 of
teacher/actor rollouts (DIR/execution/trajectory.npz or DIR/evaluation/execution/trajectory.npz, or a trajectory.npz path); shipped meshes per EP."""
import sys, numpy as np, glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import blib as B
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
MD = {32: f'{C}/R_ep32_regrasp_power06/episode_000032/meshes/episode_000032', 33: f'{C}/P_ep33_regrasp_cem3/episode_000033/meshes/episode_000033'}
BINS = '--bins' in sys.argv; sys.argv = [x for x in sys.argv if x != '--bins']
ep = int(sys.argv[1]); reg = B.register(ep, {n: Path(MD[ep]) / f'{n}.obj' for n in B.NAMES})
def load(p):
    p = Path(p)
    for c in (p, p / 'execution/trajectory.npz', p / 'evaluation/execution/trajectory.npz'):
        if c.is_file(): return c
for d in sys.argv[2:]:
    f = load(d)
    if f is None: print(d, 'missing'); continue
    z = np.load(f); names = [str(x) for x in z['object_names']]; P = {}
    for n in B.NAMES:
        i = names.index(n); T = np.tile(np.eye(4), (len(z['object_positions']), 1, 1))
        T[:, :3, :3] = R.from_quat(z['object_quaternions_wxyz'][:, i][:, [1, 2, 3, 0]]).as_matrix(); T[:, :3, 3] = z['object_positions'][:, i]; P[n] = T
    e = B.per_frame(ep, P, reg); w = max(max(e[n][0].mean() / 3, e[n][1].mean() / 30) for n in B.NAMES)
    if BINS:
        for s0 in range(0, len(e['blue_cup'][0]), 10): print(f'   {s0/20:5.1f}s', ' '.join(f"{e[n][0][s0:s0+10].mean():5.2f}/{e[n][1][s0:s0+10].mean():5.1f}" for n in B.NAMES))
    print(f'{w:5.2f}', ' '.join(f'{n[:5]} {e[n][0].mean():5.2f}/{e[n][1].mean():5.1f}' for n in B.NAMES), d, flush=True)

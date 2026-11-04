"""ep25: target_final_v5 board tracks -> OUR slotted board body frames (scorer registration, MP-SR._register_object_body, seed
1_000_003*25+101*slot, our submitted meshes = streamQ meshes_slotted_C) -> our physics world (worldC/c1 frame) by ONE rigid transform
G = Gt @ G0: G0 maps the target piece_1 (slot 0 = anchor) frame-0 pose onto the physics piece_1 frame-0 pose of worldC/c1; Gt =
table correction (rotation of the mean resting-face normal f0-30 -- face normal in body frame taken from the flat-resting worldC/c1 boards -- onto +z, vertical shift putting the lower resting board's collision 0.2 mm above z=0,
as streamQ compose_to_world.py). usage: conv25.py OUT.npz  (prints diagnostics; lineage json next to OUT)"""
import sys, json, glob, hashlib, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30/code')
from elib import *
Q = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25')
NAMES = ['wooden_piece_1', 'wooden_piece_2']
MESH = {n: Q / f'meshes_slotted_C/{n}_visual_slotted.obj' for n in NAMES}
CREF = Q / 'worldC/c1/source_corrected/episode_000025/object_poses.npz'
V21 = Q / 'world/w4/source_corrected/episode_000025/object_poses.npz'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
def build():
    X = {}; Tob = []
    for slot, n in enumerate(NAMES):
        X[n], *_ = registration(25, slot, n, MESH[n]); Tob.append(target(25, n) @ X[n])
    Tob = np.stack(Tob, 1); C = np.load(CREF)['world_from_object']
    G0 = C[0, 0] @ np.linalg.inv(Tob[0, 0]); T = np.einsum('ij,tojk->toik', G0, Tob)
    coll = {o: np.vstack([trimesh.load(f, force='mesh').vertices for f in sorted(glob.glob(str(Q / f'collision_C/{o}_cutpz/part_*.obj')))]) for o in NAMES}
    # resting-face normal in body frame = world z seen from the boards SETTLED on the physics table (C_ep25_slotC1 achieved, frames 5-15;
    # the worldC/c1 reference itself is 4.3 deg off flat)
    A, _ = results_tracks(next(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/C_ep25_slotC1/episode_000025/results').glob('*.parquet')))
    nb = []
    for o in NAMES:
        v = np.mean([A[o][k, :3, :3].T @ np.array([0, 0, 1.]) for k in range(5, 16)], 0); nb.append(v / np.linalg.norm(v))
    nrm = [T[k, i, :3, :3] @ nb[i] for i in range(2) for k in range(31)]
    n = np.mean(nrm, 0); n /= np.linalg.norm(n); ax = np.cross(n, [0, 0, 1.]); s = np.linalg.norm(ax)
    Rt = R.from_rotvec(ax / s * np.arctan2(s, n[2])).as_matrix() if s > 1e-12 else np.eye(3)
    piv = T[0, 0, :3, 3]; Tt = np.eye(4); Tt[:3, :3] = Rt; Tt[:3, 3] = piv - Rt @ piv          # rotate about the anchor board
    T1 = np.einsum('ij,tojk->toik', Tt, T)
    lows = {o: float(np.median([(coll[o] @ T1[k, i, :3, :3].T + T1[k, i, :3, 3])[:, 2].min() for k in range(31)])) for i, o in enumerate(NAMES)}
    Tt[2, 3] -= min(lows.values()) - 0.0002   # lowest resting board just touches the table (+0.2 mm), the other one drops
    G = Tt @ G0; Tf = np.einsum('ij,tojk->toik', G, Tob)
    info = dict(G0=G0.tolist(), Gt=Tt.tolist(), G=G.tolist(), table_tilt_deg=float(np.degrees(np.arctan2(s, n[2]))), resting_lowest_before_cm={k: v * 100 for k, v in lows.items()},
                resting_lowest_after_cm={o: float(np.median([(coll[o] @ Tf[k, i, :3, :3].T + Tf[k, i, :3, 3])[:, 2].min() for k in range(31)]) * 100) for i, o in enumerate(NAMES)},
                X_ourbody_from_targetbody={k: v.tolist() for k, v in X.items()},
                sources={o: dict(track=f'{TGT}/episode_000025/{o}_trajectory.npz', sha256=sha(f'{TGT}/episode_000025/{o}_trajectory.npz'), target_mesh_sha256=sha(f'{TGT}/episode_000025/{o}_mesh_metric.obj'), our_mesh=str(MESH[o]), our_mesh_sha256=sha(MESH[o])) for o in NAMES})
    for lab, ref in [('C_world', C), ('v21_w4', np.load(V21)['world_from_object'])]:
        info[f'shift_vs_{lab}'] = {o: dict(zip(['mean_cm', 'mean_deg', 'per_sec_cm', 'per_sec_deg'], (lambda p, d: [float(p.mean()), float(d.mean()), [round(float(p[s*20:(s+1)*20].mean()), 2) for s in range(18)], [round(float(d[s*20:(s+1)*20].mean()), 1) for s in range(18)]])(*errs(Tf[:, i], ref[:, i])))) for i, o in enumerate(NAMES)}
    return Tf, info
if __name__ == '__main__':
    T, info = build(); out = Path(sys.argv[1]); np.savez(out, world_from_object=T, names=np.array(NAMES), G=np.array(info['G']))
    json.dump(info, open(out.with_suffix('.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ('G0', 'Gt', 'G', 'X_ourbody_from_targetbody', 'sources')}, indent=0)[:3000])

"""Floor estimate per table-plane choice: 'ideal physical rollout' = compose-derived poses with every resting frame projected onto
the plane (minimal rotation making the object's rest up-axis parallel to the plane normal, about its lowest point, then lifted /
lowered onto the plane); carried frames = compose + linear blend of the rest corrections at lift and set-down. Scored with uscore."""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent)); import uscore as U
CR = Path('/data1/SRL_V2D/challenge_reconstruction'); SRC = CR / 'final_recon/episode_000008'
S = np.array(json.load(open(CR / 'campaign_v16_20261023/streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008/input_manifest.json'))['setup_from_source'])
ours = np.load(SRC / 'object_poses.npz')['world_from_object'].astype(float); new = np.load(sys.argv[1])['world_from_object'].astype(float)
L = np.einsum('ij,tkjl->tkil', S, new)   # level frame
names = U.NAMES; V = [np.asarray(trimesh.load(SRC / f'assets/{n}.obj', force='mesh').vertices) for n in names]
# rest up-axis in body frame = minus the outward normal of the convex-hull support facet (cup: bottom disc; pot: flat bottom)
UP = [-np.array([0.005, -0.005, 1.]), -np.array([-0.986, -0.022, -0.167])]; UP = [u / np.linalg.norm(u) for u in UP]
# rest masks from compose heights: within 1 cm of the frame-0 or last-frame height
z = L[:, :, 2, 3]; rest = (np.abs(z - z[:1]) < .01) | (np.abs(z - z[-1:]) < .01)
for k in range(2):  # keep only the leading and trailing runs
    r = rest[:, k]; a = np.argmin(r) if not r.all() else len(r); b = len(r) - np.argmin(r[::-1]) if not r.all() else 0; m = np.zeros_like(r); m[:a] = True; m[b:] = True; rest[:, k] = m
print('rest windows (s): cup 0-%.2f, %.2f-end; pot 0-%.2f, %.2f-end' % (np.argmin(rest[:, 0]) / 20, (len(z) - np.argmin(rest[::-1, 0])) / 20, np.argmin(rest[:, 1]) / 20, (len(z) - np.argmin(rest[::-1, 1])) / 20))
def project(T, k, n, h):
    up = T[:3, :3] @ UP[k]; Rm = R.align_vectors([n], [up])[0].as_matrix()
    pts = V[k] @ T[:3, :3].T + T[:3, 3]; lo = pts[np.argmin(pts @ n)]
    T2 = T.copy(); T2[:3, :3] = Rm @ T[:3, :3]; T2[:3, 3] = Rm @ (T[:3, 3] - lo) + lo
    pts2 = V[k] @ T2[:3, :3].T + T2[:3, 3]; T2[:3, 3] += n * (h - (pts2 @ n).min()); return T2
def ideal(n, h):
    n = n / np.linalg.norm(n); P = L.copy()
    for k in range(2):
        idx = np.flatnonzero(rest[:, k]); a = np.argmin(rest[:, k]); b = len(z) - np.argmin(rest[::-1, k])
        for t in idx: P[t, k] = project(L[t, k], k, n, h)
        Da = P[a - 1, k] @ np.linalg.inv(L[a - 1, k]); Db = P[b, k] @ np.linalg.inv(L[b, k])
        for t in range(a, b):
            w = (t - a) / max(1, b - a); da = Da[:3, 3] * (1 - w) + Db[:3, 3] * w; P[t, k, :3, 3] = L[t, k, :3, 3] + da
    return P
n_pot = L[0, 1, :3, :3] @ UP[1]; n_cup = L[0, 0, :3, :3] @ UP[0]
def pot_h(n):
    pts = V[1] @ L[0, 1, :3, :3].T + L[0, 1, :3, 3]; return (pts @ (n / np.linalg.norm(n))).min()
def cup_h(n):
    pts = V[0] @ L[0, 0, :3, :3].T + L[0, 0, :3, 3]; return (pts @ (n / np.linalg.norm(n))).min()
print('tilt pot-normal vs level %.1f deg, cup-normal %.1f deg, pot-vs-cup normals %.1f deg' % (np.degrees(np.arccos(n_pot[2])), np.degrees(np.arccos(n_cup[2])), np.degrees(np.arccos(n_pot @ n_cup))), 'n_pot', np.round(n_pot, 3), 'n_cup', np.round(n_cup, 3))
res = {}
for lab, n in [('level', np.array([0, 0, 1.])), ('pot', n_pot), ('mid', n_pot + np.array([0, 0, 1.]) * 0 + n_cup), ('pot+level', n_pot + np.array([0, 0, 1.]))]:
    n = n / np.linalg.norm(n)
    for hl, h in [('pot', pot_h(n)), ('cup', cup_h(n)), ('avg', (pot_h(n) + cup_h(n)) / 2)]:
        o = U.score(ideal(n, h)); res[f'{lab}/{hl}'] = o; print('%-10s h=%-4s cup-pot height gap %.2f cm' % (lab, hl, (cup_h(n) - pot_h(n)) * 100), o)
# two-object plane: normal tilted from n_pot so both bottoms touch
best = None
for ax in np.linspace(-1, 1, 21):
    for ay in np.linspace(-1, 1, 21):
        n = n_pot + np.array([ax, ay, 0]) * .15; n /= np.linalg.norm(n); h = (pot_h(n) + cup_h(n)) / 2
        o = U.score(ideal(n, h)); s = max(o['beige_cup'][0] / 2.4, o['white_pot'][0] / 2.4, o['beige_cup'][1] / 24, o['white_pot'][1] / 24)
        if best is None or s < best[0]: best = (s, n, h, o)
print('best plane search: ratio %.3f n %s tilt %.1f deg h %.4f' % (best[0], np.round(best[1], 4), np.degrees(np.arccos(best[1][2])), best[2]), best[3])
json.dump(dict(n_pot=n_pot.tolist(), best_n=best[1].tolist(), best_h=float(best[2]), best_scores=best[3], options=res), open(Path(sys.argv[1]).parent / 'plane_options.json', 'w'), indent=1)

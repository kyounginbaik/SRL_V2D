"""K33-recon2 (GT-free): spoon roll scan with the TWO-SIDED occlusion-aware cost of evid.py (best over small image shifts), per frame.
usage: rollscan2.py BUNDLE_EP_DIR OUT.tsv STEP [MESH]"""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as Rot
B, OUT, STEP = sys.argv[1], sys.argv[2], int(sys.argv[3]); MESH = sys.argv[4] if len(sys.argv) > 4 else None
sys.argv_saved = OUT; sys.argv = ["evid", B, 'wooden_spoon', '/dev/null'] + ([MESH] if MESH else [])
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evid.py')).read().split("if __name__")[0])
OUT = sys.argv_saved
com = np.asarray(m.center_mass); c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
if long[np.argmax(np.abs(long))] < 0: long = -long
ROLLS = [-45, -30, -20, -10, -5, 0, 5, 10, 20, 30, 45]
SH2 = np.arange(-15, 16, 3)
def cost(t, T, M, hand, D):
    zb = render(T); best = 9
    for du in SH2:
        for dvv in SH2:
            zs = np.roll(np.roll(zb, dvv, 0), du, 1); rm = np.isfinite(zs); occ = hand | (np.isfinite(D) & (D < zs - 0.015))
            best = min(best, ((M & ~rm).sum() + (rm & ~M & ~occ).sum()) / M.sum())
    return best
def one(t):
    M = rd(f'{C}/objects/{OBJ}/masks/1/{t:06d}.png') > 0
    hand = (rd(f'{C}/objects/{OBJ}/masks/2/{t:06d}.png') > 0) | (rd(f'{C}/objects/{OBJ}/masks/3/{t:06d}.png') > 0) | ((rd(f'{C}/objects/{OTHER}/masks/1/{t:06d}.png') > 0) if OTHER else False)
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    if M.sum() < 300: return [t, int(M.sum())] + [np.nan] * len(ROLLS)
    out = [t, int(M.sum())]
    for r in ROLLS:
        E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); Bm = np.eye(4); Bm[:3, :3] = E; Bm[:3, 3] = com - E @ com
        out.append(cost(t, Tc[t] @ Bm, M, hand, D))
    return out
if __name__ == '__main__':
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, range(0, len(Tc), STEP))
    open(OUT, 'w').write('frame\tmaskpx\t' + '\t'.join(f'r{r}' for r in ROLLS) + '\n' + '\n'.join('\t'.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in r) for r in rows) + '\n')
    print('done', len(rows))

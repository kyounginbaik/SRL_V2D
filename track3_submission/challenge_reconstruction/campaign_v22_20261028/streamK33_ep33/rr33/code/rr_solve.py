"""RR33 Viterbi over rr_scan output. State = roll grid. Unary = sil + 0.3*ball(px/20) + WD*depth(mm/10) + weak roll prior. Pairwise = |d roll| + excess world rotation step over the base track + spoon-in-wrist change (only if a wrist prior exists).
Frames outside [a,b] (first/last motion +- margin, or explicit) stay exactly base; ends pinned to roll 0. Tilt and shift taken from the selected state and low-passed.
usage: rr_solve.py SCAN.npz EP A B OUT.npz [WD=0.1] [LAM=0.03] [LW=0.03] [LH=0.02]"""
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d as gf
scan, EP, A, Bb, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
WD, LAM, LW, LH = [float(sys.argv[6+i]) if len(sys.argv) > 6+i else d for i, d in enumerate((0.1, 0.03, 0.03, 0.02))]
z = np.load(scan, allow_pickle=False); R0 = z['rolls']; S = z['sil']; n = len(R0); N = len(S); Rw = z['Rw'].astype(np.float64); i0 = int(np.argmin(np.abs(R0)))
def norm(x, cap): return np.nan_to_num(np.minimum(x, cap)/cap, nan=0.0)
U = np.nan_to_num(S, nan=0.0) + 0.3*norm(z['ball'], 20.) + WD*norm(z['dep'], 10.) + 0.02*np.abs(R0)/30.
wf = None; H = f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EP:06d}/hand_motion_prior_v2.npz'
try:
    h = np.load(H, allow_pickle=False); k = [str(x) for x in h['names']].index('wooden_spoon'); wf = h['world_from_wrist'][:, k]; wv = h['wrist_valid'][:, k] & h['held'][:, k]
except FileNotFoundError: pass
def ang(Ra, Rb):  # all pairs angle deg (n,3,3),(n,3,3)
    tr = np.einsum('aij,bij->ab', Ra, Rb); return np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1)))
base = np.array([Rw[t, i0] for t in range(N)])  # roll 0, best tilt: reference step
bstep = np.r_[[np.degrees(Rot.from_matrix(base[t+1]@base[t].T).magnitude()) for t in range(N-1)], 0]
C = U[A].copy() + 100*(np.abs(R0) > 3); bp = np.zeros((N, n), int)
dR = np.abs(R0[:, None]-R0[None])/10.
for t in range(A+1, Bb+1):
    W = ang(Rw[t-1], Rw[t]); P = LAM*dR + LW*np.maximum(W - bstep[t-1] - 5, 0)/10 + 5.0*(W > 45)
    if wf is not None and wv[t-1] and wv[t]:
        Ra = np.einsum('ji,ajk->aik', wf[t-1,:3,:3], Rw[t-1]); Rb = np.einsum('ji,ajk->aik', wf[t,:3,:3], Rw[t]); P = P + LH*ang(Ra, Rb)/10
    c = C[:, None] + P; bp[t] = np.argmin(c, 0); C = c[bp[t], np.arange(n)] + U[t]
C = C + 100*(np.abs(R0) > 3); path = np.full(N, i0); path[Bb] = int(np.argmin(C))
for t in range(Bb, A, -1): path[t-1] = bp[t, path[t]]
idx = np.arange(N); roll = R0[path].copy(); roll[:A] = 0; roll[Bb+1:] = 0
tilt = z['tilt'][idx, path]; shift = z['shift'][idx, path]; tilt[:A] = 0; tilt[Bb+1:] = 0; shift[:A] = 0; shift[Bb+1:] = 0
win = np.zeros(N); win[A:Bb+1] = 1
s = lambda x, sg: gf(x, sg, mode='nearest') * gf(win, 1.5, mode='nearest').clip(0, 1)
roll_s = gf(roll, 1.0, mode='nearest'); tilt_s = np.c_[gf(tilt[:, 0], 2.0, mode='nearest'), gf(tilt[:, 1], 2.0, mode='nearest')]; shift_s = np.c_[gf(shift[:, 0], 1.5, mode='nearest'), gf(shift[:, 1], 1.5, mode='nearest')]
for a in (roll_s, tilt_s, shift_s): a[:A] = 0; a[Bb+1:] = 0
np.savez(OUT, roll=roll_s, tilt=tilt_s, shift=shift_s, raw_roll=roll, A=A, B=Bb, params=np.array([WD, LAM, LW, LH]))
print(f'ep{EP} range {A}-{Bb} wrist={wf is not None}: mean|roll| in range {np.abs(roll_s[A:Bb+1]).mean():.1f}, mean|mid| {np.abs(tilt_s[A:Bb+1,0]).mean():.1f}, mean|short| {np.abs(tilt_s[A:Bb+1,1]).mean():.1f}, shift px {np.linalg.norm(shift_s[A:Bb+1],axis=1).mean():.1f}, max step roll {np.abs(np.diff(roll_s)).max():.1f}')
print(' roll per s: '+' '.join(f'{s_//20}s:{roll_s[s_:s_+20].mean():+.0f}' for s_ in range(0,N,20)))
sil_b=np.nan_to_num(z['sil00']); sil_a=np.nan_to_num(S[idx,path]); print(' sil cost per s base->sol: '+' '.join(f'{s_//20}s:{sil_b[s_:s_+20].mean():.2f}>{sil_a[s_:s_+20].mean():.2f}' for s_ in range(0,N,20)))

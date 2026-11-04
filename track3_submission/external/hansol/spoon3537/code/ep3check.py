"""ep3 FINAL recon check, both objects, per 0.5 s: canonical (stereo) camera IoU vs canonical masks + marker-ball reprojection
(median of per-detection nearest model ball, px, capped 30) vs the RGB path (SAM2 masks, world_from_rgb, undistorted K). GT-free."""
import sys, json, numpy as np, cv2, trimesh
from pathlib import Path
sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code'); sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import DELTA
from guard_clip import Evidence, CAN
from retime import shift
EP = 3; E = 'episode_000003'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
BUN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17') / E
TG = C22 / 'target_final_v14_g17' / E; COMP = C22 / 'streamALL20K/staging/tgt_v32_ep3_b2/components' / E
OBJ = ['white_pot', 'wooden_spoon']
z = np.load(BUN / 'object_poses.npz'); bn = [str(n) for n in z['names']]; F = len(z['camera_from_object'])
root = CAN / 'evaluation' / E; dd = DELTA[f'evaluation_{EP}']; dd = np.asarray(dd, float)[:F] if np.ndim(dd) else np.full(F, float(dd))
I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
mj = json.loads((BUN / 'meshes.json').read_text()); ev = Evidence(root, OBJ, {o: BUN / mj[o]['file'] for o in OBJ}, F)
BM = json.loads((root / 'ball_models_v1.json').read_text())
man = json.loads((COMP / f'portable_validation/runtime/episodes/{E}/input_manifest.json').read_text()); SRC = Path(man['source'])
op = np.load(SRC / 'object_poses.npz'); sn = [str(n) for n in op['names']]; Wrgb = op['world_from_rgb'].astype(np.float64)
SAMO = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs') / E
res = {}
for o in OBJ:
    Tt = np.load(TG / f'{o}_trajectory.npz')['T_world_obj']; Wb = z['world_from_object'][:, bn.index(o)]
    camC = shift(z['camera_from_object'][:, bn.index(o)], -dd)
    print(o, 'target vs bundle world (max |diff|)', float(np.abs(Tt - Wb).max()))
    iouC = ev.eval(o, camC, list(range(F)))[0]
    B = np.asarray(BM[o]) if o in BM else None; DET = json.loads((root / f"ball_detections_{o}.json").read_text())["frames"]
    Vb = np.asarray(trimesh.load(BUN / mj[o]["file"], force="mesh", process=False).vertices)
    def bp(T, det):
        if not det: return np.nan
        if B is None:   # no ball model: 3D distance (cm) from detected ball centres to the nearest mesh vertex, capped 5 cm
            P = Vb @ T[:3, :3].T + T[:3, 3]; return float(np.median([min(5.0, 100 * np.min(np.linalg.norm(P - np.asarray(dd_["camera_point"]), axis=1))) for dd_ in det]))
        p = B @ T[:3, :3].T + T[:3, 3]; u = p[:, :2] / p[:, 2:3] @ K[:2, :2].T + K[:2, 2]
        c = T[:3, 3]; det = [d for d in det if 'camera_point' not in d or np.linalg.norm(np.asarray(d['camera_point']) - c) < 0.25]
        return float(np.median([min(30, np.min(np.linalg.norm(u - np.asarray(d['uv']), axis=1))) for d in det])) if det else np.nan
    ball = np.array([bp(camC[f], DET[f] if f < len(DET) else []) for f in range(F)])
    # RGB path
    Ws = op['world_from_object'][:, sn.index(o)].astype(np.float64); camR = np.einsum('tij,tjk->tik', np.linalg.inv(Wrgb), Ws)
    m = trimesh.load(SRC / f'assets/{o}.obj', force='mesh', process=False); V = np.asarray(m.vertices); Fc = np.asarray(m.faces)
    d = json.loads((SAMO / o / 'anycalib/undistorted_intrinsics.json').read_text()); SC = 0.5; Kr = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1.]]) * [[SC], [SC], [1]]
    aw, ah = int(d['width'] * SC), int(d['height'] * SC); iouR = np.full(F, np.nan); cenR = np.full(F, np.nan)
    for t in range(F):
        try:
            rd = lambda k: cv2.resize(cv2.imread(str(SAMO / o / f'masks/{k}/{t:06d}.png'), cv2.IMREAD_GRAYSCALE), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
            Mk = rd(1); h = rd(2) | rd(3)
        except Exception: continue
        P = V @ camR[t][:3, :3].T + camR[t][:3, 3]
        if (P[:, 2] < 0.02).any() or (Mk & ~h).sum() < 150: continue
        uv = P[:, :2] / P[:, 2:3] @ Kr[:2, :2].T + Kr[:2, 2]; S = np.zeros((ah, aw), np.uint8); cv2.fillPoly(S, list(np.round(uv[Fc] * 4).astype(np.int32)), 1, shift=2); S = S > 0
        a = S & ~h; b = Mk & ~h; iouR[t] = (a & b).sum() / max((a | b).sum(), 1)
        ys, xs = np.nonzero(b); ya, xa = np.nonzero(a)
        if len(xa): cenR[t] = np.hypot(xs.mean() - xa.mean(), ys.mean() - ya.mean()) / SC
    zc = np.array([np.linalg.norm(camC[f][:3, 3]) for f in range(F)])
    res[o] = (iouC, ball, iouR, cenR, zc)
print('t0  | pot: IoU stereo  ball px (~cm)  IoU rgb  cen rgb px | spoon: IoU stereo  ball px (~cm)  IoU rgb  cen rgb px')
for a in np.arange(0, F / 20, 0.5):
    f = list(range(int(a * 20), min(F, int((a + 0.5) * 20))))
    row = []
    for o in OBJ:
        iouC, ball, iouR, cenR, zc = res[o]; bm = np.nanmedian(ball[f]) if np.isfinite(ball[f]).any() else np.nan
        row.append('%.2f  %5.1f (%.1f)  %.2f  %5.1f' % (np.nanmean(iouC[f]), bm, bm * np.nanmean(zc[f]) / K[0, 0] * 100, np.nanmean(iouR[f]), np.nanmean(cenR[f])))
    print('%4.1f | %s | %s' % (a, row[0], row[1]))
for o in OBJ:
    iouC, ball, iouR, cenR, zc = res[o]; f = list(range(160, 175))
    print(o, 'f160-174 (8.0-8.75 s): IoU stereo %.2f ball %.1f px, IoU rgb %.2f; whole: IoU stereo %.2f ball median %.1f, IoU rgb %.2f' % (
        np.nanmean(iouC[f]), np.nanmedian(ball[f]), np.nanmean(iouR[f]), np.nanmean(iouC), np.nanmedian(ball), np.nanmean(iouR)))

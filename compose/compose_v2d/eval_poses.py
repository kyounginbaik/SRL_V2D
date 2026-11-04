"""Re-evaluate saved tracking results (track{tag}/poses.npz) against the GT mocap trajectory.
Shows, per variant, the GT time-offset sweep and two alignments: 'start' (the kit: one rigid transform from frame 0)
and 'all' (rigid transform fitted on all frames) -> separates the frame-0 anchor error from the tracking error.
Also compares the frame-0 anchor pose between runs (--compare_tag)."""
import argparse, os, sys
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T3, work_dir, load_meta, gt_world_T_obj
from track_compose import load_kit_consts, _safe_R, T_of


def err_tables(ours, gt, verts, consts):
    """ours, gt: (n,4,4) already in a common world frame."""
    pos = np.linalg.norm(ours[:, :3, 3] - gt[:, :3, 3], axis=1)
    ori = np.array([np.linalg.norm(Rot.from_matrix(_safe_R(ours[i, :3, :3] @ gt[i, :3, :3].T)).as_rotvec()) for i in range(len(gt))])
    pw = np.einsum('nij,vj->nvi', ours[:, :3, :3], verts) + ours[:, None, :3, 3]; pg = np.einsum('nij,vj->nvi', gt[:, :3, :3], verts) + gt[:, None, :3, 3]
    add = np.linalg.norm(pw - pg, axis=-1).mean(1)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    auc = float(trap(np.array([np.mean(add < t) for t in consts['thr']]), x=np.linspace(0, 1, len(consts['thr']))))
    return pos, ori, auc


def align_start(ours, gt):
    A = gt[0] @ np.linalg.inv(ours[0]); return np.einsum('ij,njk->nik', A, ours)


def align_all(ours, gt):
    """Rigid A minimising sum |A ours_f - gt_f| (chordal mean of gt ours^-1 rotations, then translation)."""
    Rs = np.stack([gt[i, :3, :3] @ ours[i, :3, :3].T for i in range(len(gt))])
    R = _safe_R(Rs.mean(0))
    t = (gt[:, :3, 3] - np.einsum('ij,nj->ni', R, ours[:, :3, 3])).mean(0)
    return np.einsum('ij,njk->nik', T_of(R, t), ours)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True); ap.add_argument('--tag', default='_team')
    ap.add_argument('--compare_tag', default=None); ap.add_argument('--variants', default='icp_track_6dof,fused_predict_correct,compose_reanchor_tmesh,compose_anchorK_smooth_tmesh')
    a = ap.parse_args()
    wd = work_dir(a.ep, a.obj); td = wd / f'track{a.tag}'; z = np.load(td / 'poses.npz')
    gt_T, _ = gt_world_T_obj(a.ep, a.obj); consts = load_kit_consts()
    gt_tri = trimesh.load(str(T3 / f'public/mesh/{a.obj}/{a.obj}.glb'), force='mesh')
    verts = np.asarray(gt_tri.vertices)[np.random.default_rng(0).choice(len(gt_tri.vertices), 2000, replace=False)]
    Tos = z['T_ours_from_scan']; n = len(z['E'])
    for v in a.variants.split(','):
        if v not in z.files: continue
        ours_all = np.einsum('nij,jk->nik', z[v], Tos)
        print(f'\n[{v}]  d: start-aligned pos/ori/AUC   |  all-frame-aligned pos/ori/AUC')
        for d in range(-3, 4):
            idx = [f for f in range(n) if 0 <= f + d < len(gt_T)]
            o = ours_all[idx]; g = gt_T[[f + d for f in idx]]
            ps, os_, as_ = err_tables(align_start(o, g), g, verts, consts); pa, oa, aa = err_tables(align_all(o, g), g, verts, consts)
            print(f'   d={d:+d}: {ps.mean()*100:5.2f} cm {np.degrees(os_).mean():6.2f} deg {as_:.3f}   |  {pa.mean()*100:5.2f} cm {np.degrees(oa).mean():6.2f} deg {aa:.3f}'
                  f'   (ori p50/p90 start {np.degrees(np.percentile(os_,50)):.1f}/{np.degrees(np.percentile(os_,90)):.1f}, all {np.degrees(np.percentile(oa,50)):.1f}/{np.degrees(np.percentile(oa,90)):.1f})')
    print(f'\nframe-0 anchor (scan frame): residual mesh fit scale {float(z["scale"]):.4f}')
    if a.compare_tag:
        z2 = np.load(wd / f'track{a.compare_tag}' / 'poses.npz')
        T1 = T_of(z['R0'], z['t0']) @ Tos; T2 = T_of(z2['R0'], z2['t0']) @ z2['T_ours_from_scan']
        dR = np.degrees(np.linalg.norm(Rot.from_matrix(_safe_R(T1[:3, :3] @ T2[:3, :3].T)).as_rotvec())); dt = np.linalg.norm(T1[:3, 3] - T2[:3, 3]) * 100
        rv = Rot.from_matrix(_safe_R(T2[:3, :3].T @ T1[:3, :3])).as_rotvec()
        print(f'anchor difference {a.tag} vs {a.compare_tag}: {dR:.2f} deg, {dt:.2f} cm; axis in scan frame {np.round(rv / max(np.linalg.norm(rv), 1e-9), 2)}')


if __name__ == '__main__':
    main()

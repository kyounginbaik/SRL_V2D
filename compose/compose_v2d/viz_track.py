"""Overlay the tracked metric mesh on the rect-left mono frames (and the RGB frames) for a tracking variant.
Writes outputs/compose_ep{ep}_{obj}/track{tag}/viz_{variant}.jpg (contact sheet) and .mp4.
"""
import argparse, os, sys
import cv2, numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import work_dir, load_meta, gt_world_T_obj, gt_mesh_path


def draw_mesh(img, V, F, K, T_cam_obj, color, thick=1):
    P = V @ T_cam_obj[:3, :3].T + T_cam_obj[:3, 3]
    if (P[:, 2] <= 0.05).any():
        return img
    uv = (P[:, :2] / P[:, 2:3]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    uv = np.round(uv).astype(np.int32)
    H, W = img.shape[:2]
    for f in F:
        pts = uv[f]
        if (pts[:, 0] < -W).any() or (pts[:, 0] > 2 * W).any() or (pts[:, 1] < -H).any() or (pts[:, 1] > 2 * H).any():
            continue
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, thick, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True); ap.add_argument('--tag', default='_team')
    ap.add_argument('--variants', default='fused_predict_correct,icp_track_6dof,compose_anchorK_smooth_tmesh')
    ap.add_argument('--step', type=int, default=1); ap.add_argument('--sheet_every', type=int, default=25)
    ap.add_argument('--gt', action='store_true', help='also draw the GT scan at the mocap pose, aligned to our frame-0 pose (red)')
    a = ap.parse_args()
    wd = work_dir(a.ep, a.obj); meta = load_meta(a.ep, a.obj); td = wd / f'track{a.tag}'
    z = np.load(td / 'poses.npz'); E = z['E']
    K_rect = np.array(meta['K_rect']); K_und = np.array(meta['K_a_undist']); T_a = np.array(meta['T_a_from_rectL'])
    m = trimesh.load(td / 'mesh_metric.obj', force='mesh')
    try:
        ms = m.simplify_quadric_decimation(face_count=1200)
    except Exception:
        ms = m
    V, F = np.asarray(ms.vertices), np.asarray(ms.faces)
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255)]
    variants = [v for v in a.variants.split(',') if v in z.files]
    gt_T = None
    if a.gt:
        gt_T, _ = gt_world_T_obj(a.ep, a.obj); gm = trimesh.load(gt_mesh_path(a.obj), force='mesh')
        try: gm = gm.simplify_quadric_decimation(face_count=1200)
        except Exception: pass
        Vg, Fg = np.asarray(gm.vertices), np.asarray(gm.faces)
        T0 = z[variants[0]][0] @ z['T_ours_from_scan']                     # L0_T_scan at frame 0 (our anchor + body registration)
        T_L0_world = T0 @ np.linalg.inv(gt_T[0])
        print('GT drawn in red: scan mesh at mocap pose, aligned on our frame-0 pose')
    print('variants:', variants)
    n = len(E); sheet = []
    vw = cv2.VideoWriter(str(td / 'viz.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280 + 1014, 800))
    for f in range(0, n, a.step):
        mono = cv2.cvtColor(cv2.imread(str(wd / 'mono' / f'{f:05d}.jpg'), 0), cv2.COLOR_GRAY2BGR)
        rgb = cv2.resize(cv2.imread(str(wd / 'rgb_undist' / f'{f:05d}.jpg')), (1014, 760)); K_h = K_und.copy(); K_h[:2] /= 2
        E_inv = np.linalg.inv(E[f])
        for vi, v in enumerate(variants):
            T_L0_obj = z[v][f]; T_Lf_obj = E_inv @ T_L0_obj
            draw_mesh(mono, V, F, K_rect, T_Lf_obj, colors[vi % 3])
            draw_mesh(rgb, V, F, K_h, T_a @ T_Lf_obj, colors[vi % 3])
            cv2.putText(mono, v, (10, 30 + 28 * vi), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[vi % 3], 2)
        if gt_T is not None and f < len(gt_T):
            T_Lf_scan = E_inv @ T_L0_world @ gt_T[f]
            draw_mesh(mono, Vg, Fg, K_rect, T_Lf_scan, (0, 0, 255)); draw_mesh(rgb, Vg, Fg, K_h, T_a @ T_Lf_scan, (0, 0, 255))
            cv2.putText(mono, 'GT (aligned @ frame 0)', (10, 30 + 28 * len(variants)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(mono, f'frame {f}', (10, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        canvas = np.zeros((800, 1280 + 1014, 3), np.uint8); canvas[:, :1280] = mono; canvas[20:780, 1280:] = rgb
        vw.write(canvas)
        if f % a.sheet_every == 0:
            sheet.append(cv2.resize(canvas, (canvas.shape[1] // 2, 400)))
    vw.release()
    rows = [np.concatenate(sheet[i:i + 2], 1) if i + 1 < len(sheet) else np.concatenate([sheet[i], np.zeros_like(sheet[i])], 1) for i in range(0, len(sheet), 2)]
    cv2.imwrite(str(td / 'viz_sheet.jpg'), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print('wrote', td / 'viz.mp4', td / 'viz_sheet.jpg')


if __name__ == '__main__':
    main()

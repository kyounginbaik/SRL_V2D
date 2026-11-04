"""WiLoR on the undistorted RGB frames of a prepared episode -> wilor_hands.npz (ComPose format).

Adapted from ComPose_final/internet_demo/run_wilor.py: YOLO hand detection, manipulating-hand selection by
object-mask IoU on the first frame + temporal lock afterwards, WiLoR regression, temporal jump filter.
Runs from wilor_work/ (ComPose WiLoR code + /jisu/WiLoR weights + chumpy-free MANO).
Output keys: frames, hand_pose (48 axis-angle + 3 cam t), joints_3d (21,3), joints_2d (21,2 px), bbox, is_right, focal_length.
"""
import argparse, glob, os, sys
import numpy as np, cv2, torch
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

ROOT = '/jisu/NVIDIA_CHALLENGE_claude'
WORK = os.path.join(ROOT, 'wilor_work')
sys.path.insert(0, WORK); os.chdir(WORK)
os.environ.setdefault('WILOR_SKIP_RENDERER', '1'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})   # torch>=2.6 default breaks lightning/ultralytics ckpts
import types
class _StubModule(types.ModuleType):                                   # renderer is skipped; pyrender/pyglet need X11 here
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return type(name, (), {})
sys.modules['pyrender'] = _StubModule('pyrender')

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from ultralytics import YOLO


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - img_w / 2.) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - img_h / 2.) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--jump_thresh_px', type=float, default=250.0); ap.add_argument('--jump_window', type=int, default=11)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--hand', default='auto', choices=['auto', 'right', 'left'], help='auto = manipulating hand (object IoU, then nearest to previous); right/left = only that side')
    ap.add_argument('--out_name', default='wilor_hands.npz')
    a = ap.parse_args()
    device = torch.device(a.device)
    wd = os.path.join(ROOT, 'outputs', f'compose_ep{a.ep}_{a.obj}')
    img_paths = sorted(glob.glob(os.path.join(wd, 'rgb_undist', '*.jpg')))
    print(f'WiLoR on {len(img_paths)} frames from {wd}')

    model, cfg = load_wilor('pretrained_models/wilor_final.ckpt', 'pretrained_models/model_config.yaml')
    detector = YOLO('pretrained_models/detector.pt')
    model = model.to(device).eval(); detector = detector.to(device)

    results = {}; prev_center = None
    for fi, ip in enumerate(tqdm(img_paths)):
        img = cv2.imread(ip)
        if img is None: continue
        Hi, Wi = img.shape[:2]
        mp = os.path.join(wd, 'masks_rgb', f'{fi:05d}.png')
        m_obj = (cv2.imread(mp, 0) > 0) if os.path.exists(mp) else None
        dets = detector(img, conf=a.conf, verbose=False)[0]
        bbs, irs = [], []
        for d in dets:
            try:
                irs.append(float(d.boxes.cls.cpu().item()))
                bbs.append(d.boxes.data.cpu().numpy().flatten()[:4].tolist())
            except Exception:
                continue
        if a.hand != 'auto':
            want = 1.0 if a.hand == 'right' else 0.0
            keep_ = [k for k in range(len(bbs)) if irs[k] == want]; bbs = [bbs[k] for k in keep_]; irs = [irs[k] for k in keep_]
        if not bbs: continue
        bbs_arr = np.array(bbs, np.float32)
        centers = np.stack([(bbs_arr[:, 0] + bbs_arr[:, 2]) / 2, (bbs_arr[:, 1] + bbs_arr[:, 3]) / 2], 1)
        scores = np.zeros(len(bbs))
        if m_obj is not None and m_obj.any():
            for k, bb in enumerate(bbs):
                x1, y1, x2, y2 = max(0, int(bb[0])), max(0, int(bb[1])), min(Wi, int(bb[2])), min(Hi, int(bb[3]))
                if x2 <= x1 or y2 <= y1: continue
                hb = np.zeros((Hi, Wi), bool); hb[y1:y2, x1:x2] = True
                scores[k] = (hb & m_obj).sum() / max((hb | m_obj).sum(), 1)
        if prev_center is not None:
            order = np.argsort(np.linalg.norm(centers - np.array(prev_center), axis=1)).tolist()
        elif scores.max() > 0:
            order = np.argsort(-scores).tolist()
        elif m_obj is not None and m_obj.any():
            ys, xs = np.where(m_obj); ref = np.array([xs.mean(), ys.mean()])
            order = np.argsort(np.linalg.norm(centers - ref, axis=1)).tolist()
        else:
            order = list(range(len(bbs)))
        bbs_arr = bbs_arr[order]; irs = [irs[i] for i in order]; centers = centers[order]

        ds = ViTDetDataset(cfg, img, bbs_arr, np.stack(irs), rescale_factor=2.0)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)))
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        n = 0
        ir = batch['right'][n].cpu().numpy()
        prev_center = (float(centers[0, 0]), float(centers[0, 1]))
        j3d = out['pred_keypoints_3d'][n].detach().cpu().numpy(); j3d[:, 0] = (2 * ir - 1) * j3d[:, 0]
        go = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy().reshape(-1, 3, 3)
        hp = out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy().reshape(-1, 3, 3)
        go_aa = Rot.from_matrix(go).as_rotvec().flatten(); hp_aa = Rot.from_matrix(hp).as_rotvec().flatten()
        mul = (2 * batch['right'][n] - 1).float()
        pc = out['pred_cam'][n].detach().clone(); pc[1] = mul * pc[1]
        isz = batch['img_size'][n].float()
        fl = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * isz.max()
        ct = cam_crop_to_full(pc.unsqueeze(0), batch['box_center'][n:n + 1].float(), batch['box_size'][n:n + 1].float(),
                              isz.unsqueeze(0), fl)[0].detach().cpu().numpy()
        fl_val = float(fl.cpu().numpy())
        pts_j = j3d + ct
        jx = fl_val * pts_j[:, 0] / (pts_j[:, 2] + 1e-6) + Wi / 2; jy = fl_val * pts_j[:, 1] / (pts_j[:, 2] + 1e-6) + Hi / 2
        verts = out['pred_vertices'][n].detach().cpu().numpy(); verts[:, 0] = (2 * ir - 1) * verts[:, 0]
        pv = verts + ct
        vx = fl_val * pv[:, 0] / (pv[:, 2] + 1e-6) + Wi / 2; vy = fl_val * pv[:, 1] / (pv[:, 2] + 1e-6) + Hi / 2
        results[fi] = dict(hand_pose=np.concatenate([go_aa, hp_aa, ct]).astype(np.float32), joints_3d=j3d.astype(np.float32),
                           joints_2d=np.stack([jx, jy], -1).astype(np.float32),
                           bbox=np.array([vx.min(), vy.min(), vx.max(), vy.max()], np.float32), is_right=float(ir),
                           focal_length=fl_val, obj_iou=float(scores[order[0]]))
    if not results:
        print('No hand detections'); return
    keys_all = sorted(results); centers_arr = np.stack([results[f]['joints_2d'].mean(0) for f in keys_all])
    half = a.jump_window // 2; N = len(keys_all); keep = np.ones(N, bool)
    for i in range(N):
        med = np.median(centers_arr[max(0, i - half):min(N, i + half + 1)], axis=0)
        if np.linalg.norm(centers_arr[i] - med) > a.jump_thresh_px: keep[i] = False
    if (~keep).any():
        print(f'Jump filter: dropping {int((~keep).sum())}/{N} frames')
        results = {keys_all[i]: results[keys_all[i]] for i in range(N) if keep[i]}
    keys = sorted(results)
    np.savez(os.path.join(wd, a.out_name), frames=np.array(keys, np.int64),
             hand_pose=np.stack([results[f]['hand_pose'] for f in keys]), joints_3d=np.stack([results[f]['joints_3d'] for f in keys]),
             joints_2d=np.stack([results[f]['joints_2d'] for f in keys]), bbox=np.stack([results[f]['bbox'] for f in keys]),
             is_right=np.array([results[f]['is_right'] for f in keys], np.float32),
             focal_length=np.array([results[f]['focal_length'] for f in keys], np.float32),
             obj_iou=np.array([results[f]['obj_iou'] for f in keys], np.float32))
    print(f'Saved {a.out_name}: {len(keys)}/{len(img_paths)} frames; right-hand fraction {np.mean([results[f]["is_right"] for f in keys]):.2f}')


if __name__ == '__main__':
    main()

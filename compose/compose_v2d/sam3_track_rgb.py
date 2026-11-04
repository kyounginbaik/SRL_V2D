"""SAM3 video tracking on the RGB stream (text prompt on frame 0, propagated), then transfer to the rect-left mono frame
via the stereo depth. Writes the same layout as sam3_track_mono.py (outputs/mono_masks_ep{ep}_{obj}/obj_*.png, hand_*.png)
plus the RGB masks (outputs/rgb_masks_ep{ep}_{obj}/obj_*.png, hand_*.png) at raw RGB resolution.
Colour makes small grey objects (brush, dust pan) segmentable where the mono stream fails.
usage: sam3_track_rgb.py --ep 18 --obj mini_sweeper --prompts "small hand broom" "brush"
"""
import argparse, functools, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, cam_params, load_calib, load_depth, load_T_a_from_rectL, transfer_mask, video_path
import sam3.model_builder as _mb
_mb.build_sam3_video_model = functools.partial(_mb.build_sam3_video_model, load_from_HF=False)
from sam3.model_builder import build_sam3_video_predictor


def ensure_rgb_frames(ep):
    d = ROOT / 'outputs' / f'rgb_frames_ep{ep}'
    if d.exists() and len(list(d.glob('*.jpg'))) > 0:
        return d
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path(ep, 'a'))); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        cv2.imwrite(str(d / f'{i:05d}.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY, 92]); i += 1
    print(f'wrote {i} raw RGB frames to {d}'); return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True)
    ap.add_argument('--prompts', nargs='+', required=True); ap.add_argument('--hand_prompt', default='hand')
    ap.add_argument('--min_px', type=int, default=300, help='frame-0 mask pixels needed to accept a prompt')
    ap.add_argument('--instance', default='all', choices=['all', 'left', 'right', 'largest'], help='which frame-0 instance to keep when the prompt matches several (identical objects)')
    a = ap.parse_args()
    fr_dir = ensure_rgb_frames(a.ep)
    n = len(list(fr_dir.glob('*.jpg')))
    calib = load_calib(); Ka, Da, (Wa, Ha) = cam_params(calib, 'ego_cam_a')
    depth_all, dframes, dmeta = load_depth(a.ep); K_rect = np.array(dmeta['K_rect']); T_a, _ = load_T_a_from_rectL(a.ep, dmeta)
    out_m = ROOT / 'outputs' / f'mono_masks_ep{a.ep}_{a.obj}'; out_r = ROOT / 'outputs' / f'rgb_masks_ep{a.ep}_{a.obj}'
    out_m.mkdir(exist_ok=True); out_r.mkdir(exist_ok=True)

    pred = build_sam3_video_predictor(checkpoint_path='/jisu/ObjectTracking/OnGoing/externals/sam3/checkpoints/sam3.pt')
    sid = pred.handle_request(request=dict(type='start_session', resource_path=str(fr_dir)))['session_id']

    def track(prompt):
        pred.handle_request(request=dict(type='reset_session', session_id=sid))
        r = pred.handle_request(request=dict(type='add_prompt', session_id=sid, frame_index=0, text=prompt))
        ids = r['outputs']['out_obj_ids'].tolist(); probs = np.round(np.asarray(r['outputs']['out_probs']), 2).tolist()
        m0 = np.asarray(r['outputs']['out_binary_masks']); px0 = int(m0.any(0).sum()) if len(m0) else 0
        print(f'  prompt "{prompt}": frame0 objs {ids} probs {probs} mask px {px0}', flush=True)
        if px0 < a.min_px:
            return None
        keep_id = None
        if a.instance != 'all' and len(m0) > 1 and prompt != a.hand_prompt:
            cents = [(np.nonzero(mm)[1].mean() if mm.any() else np.nan) for mm in m0]; areas = [int(mm.sum()) for mm in m0]
            sel = {'left': int(np.nanargmin(cents)), 'right': int(np.nanargmax(cents)), 'largest': int(np.argmax(areas))}[a.instance]
            keep_id = ids[sel]; print(f'  keeping instance {keep_id} ({a.instance}; centroids x = {np.round(cents, 0).tolist()})', flush=True)
        masks = {}
        for resp in pred.handle_stream_request(request=dict(type='propagate_in_video', session_id=sid)):
            o = resp['outputs']; m = np.asarray(o['out_binary_masks']); oids = np.asarray(o['out_obj_ids']).tolist()
            if keep_id is not None:
                m = m[[i for i, oid in enumerate(oids) if oid == keep_id]] if len(m) else m
            masks[resp['frame_index']] = m.any(0) if len(m) else None
        return masks

    obj = None
    for p in a.prompts:
        obj = track(p)
        if obj is not None: break
    if obj is None:
        print('no prompt produced a frame-0 mask; aborting'); sys.exit(1)
    hand = track(a.hand_prompt) or {}

    stats = []
    for i in range(n):
        om = obj.get(i); hm = hand.get(i)
        om = np.zeros((Ha, Wa), bool) if om is None else (cv2.resize(om.astype(np.uint8), (Wa, Ha), interpolation=cv2.INTER_NEAREST) > 0 if om.shape != (Ha, Wa) else om)
        hm = np.zeros((Ha, Wa), bool) if hm is None else (cv2.resize(hm.astype(np.uint8), (Wa, Ha), interpolation=cv2.INTER_NEAREST) > 0 if hm.shape != (Ha, Wa) else hm)
        cv2.imwrite(str(out_r / f'obj_{i:05d}.png'), om.astype(np.uint8) * 255); cv2.imwrite(str(out_r / f'hand_{i:05d}.png'), hm.astype(np.uint8) * 255)
        if i in dframes:
            depth = depth_all[dframes.index(i)]
            ml = transfer_mask(om, depth, K_rect, T_a, Ka, Da) if om.any() else np.zeros(depth.shape, bool)
            hl = transfer_mask(hm, depth, K_rect, T_a, Ka, Da) if hm.any() else np.zeros(depth.shape, bool)
            ml = cv2.morphologyEx(ml.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            hl = cv2.dilate(hl.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        else:
            ml = np.zeros((800, 1280), bool); hl = np.zeros((800, 1280), bool)
        cv2.imwrite(str(out_m / f'obj_{i:05d}.png'), ml.astype(np.uint8) * 255); cv2.imwrite(str(out_m / f'hand_{i:05d}.png'), hl.astype(np.uint8) * 255)
        stats.append(dict(frame=i, obj_px=int(ml.sum()), hand_px=int(hl.sum()), rgb_obj_px=int(om.sum())))
    json.dump(stats, open(out_m / 'track_stats.json', 'w'))
    empty = [s['frame'] for s in stats if s['obj_px'] == 0]
    print(f'{n} frames; rect-left object mask median {int(np.median([s["obj_px"] for s in stats]))} px; empty frames {len(empty)} {empty[:20]}')


if __name__ == '__main__':
    main()

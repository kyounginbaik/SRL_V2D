"""Write the rectified-left mono frames of an episode to outputs/mono_frames_ep{ep}/{f:05d}.jpg
(the layout sam3_track_mono.py / stereo_vo.py / prep_episode.py expect)."""
import argparse, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_calib, video_path
from rectify import EgoStereoRectifier


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int, required=True); a = ap.parse_args()
    out = ROOT / 'outputs' / f'mono_frames_ep{a.ep}'; out.mkdir(parents=True, exist_ok=True)
    rect = EgoStereoRectifier(load_calib(), episode=a.ep)
    capb = cv2.VideoCapture(str(video_path(a.ep, 'b'))); capc = cv2.VideoCapture(str(video_path(a.ep, 'c'))); i = 0
    while True:
        okb, fb = capb.read(); okc, fc = capc.read()
        if not (okb and okc): break
        L, _ = rect.rectify(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(str(out / f'{i:05d}.jpg'), L, [cv2.IMWRITE_JPEG_QUALITY, 95]); i += 1
    print(f'ep{a.ep}: wrote {i} rectified-left frames to {out}')


if __name__ == '__main__':
    main()

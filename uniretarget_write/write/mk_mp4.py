"""frames npz (from nrender.py) -> labeled mp4 at 40 fps. Usage: mk_mp4.py FRAMES.npz OUT.mp4 LABEL"""
import sys, numpy as np, cv2, imageio
d = np.load(sys.argv[1]); F = d['frames']; lab = sys.argv[3]
w = imageio.get_writer(sys.argv[2], fps=40, codec='libx264', quality=7, macro_block_size=8)
for i, f in enumerate(F):
    f = np.ascontiguousarray(f); cv2.putText(f, lab, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(f, f"t={i/40:.2f}s  (Newton)", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); w.append_data(f)
w.close(); print('wrote', sys.argv[2])

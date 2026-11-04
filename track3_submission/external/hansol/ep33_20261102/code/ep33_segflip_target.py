"""ep33 hypothesis target (GT-free): BASE target everywhere, but the spoon ROTATION taken from DONOR inside the given frame segments
(slerp ramp of RAMP frames at each edge; spoon position and the cup stay BASE). Used to test "the spoon flip is right in some segments
and wrong in others". usage: ep33_segflip_target.py BASE_EP DONOR_EP OUT_EP "a:b,c:d" RAMP"""
import sys, shutil, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rt, Slerp
base, donor, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); segs = [tuple(map(int, s.split(':'))) for s in sys.argv[4].split(',')]; rp = int(sys.argv[5])
out.mkdir(parents=True, exist_ok=False)
for p in base.iterdir():
    if not p.name.startswith('wooden_spoon_trajectory'): shutil.copy2(p, out / p.name)
Tb = np.load(base / 'wooden_spoon_trajectory.npz')['T_world_obj'].copy(); Td = np.load(donor / 'wooden_spoon_trajectory.npz')['T_world_obj']; n = min(len(Tb), len(Td))
f = np.arange(n); w = np.zeros(n)
for a, b in segs: w = np.maximum(w, np.clip(np.minimum((f - a) / rp + 1, (b - f) / rp + 1), 0, 1))
Rb, Rd = Rt.from_matrix(Tb[:n, :3, :3]), Rt.from_matrix(Td[:n, :3, :3])
for k in np.nonzero(w > 0)[0]:
    s = Slerp([0, 1], Rt.concatenate([Rb[k], Rd[k]])); Tb[k, :3, :3] = s(w[k]).as_matrix()
np.savez_compressed(out / 'wooden_spoon_trajectory.npz', T_world_obj=Tb)
(out / 'SEGFLIP.json').write_text(json.dumps(dict(base=str(base), donor=str(donor), segments=segs, ramp=rp, frames_changed=int((w > 0).sum()), ground_truth_used=False)))
print(out, 'frames changed', int((w > 0).sum()))

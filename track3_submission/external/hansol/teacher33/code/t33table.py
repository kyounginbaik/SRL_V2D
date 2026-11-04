"""T33: per-0.5 s table (errors vs $E31_TGT, finger contacts, palm azimuth) for rollouts.
usage: t33table.py SCENE|auto ROLLOUT.npz[:LABEL] ...  -> stdout (+ <ROLLOUT stem>_t33table.json next to it if writable)
ROLLOUT = trajectory.npz (key qpos) or eval.npz (key Q). SCENE auto = <rollout dir>/scene_physics.xml."""
import sys, json, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import t33lib as T
sc = sys.argv[1]
for a in sys.argv[2:]:
    f, _, lab = a.partition(':'); z = np.load(f, allow_pickle=True); Q = z['qpos'] if 'qpos' in z.files else z['Q']
    s = str(Path(f).parent / 'scene_physics.xml') if sc == 'auto' else sc; m = mujoco.MjModel.from_xml_path(s)
    res, txt = T.table(m, Q, lab or f); print(txt, flush=True)
    try: Path(f).with_name(Path(f).stem + '_t33table.json').write_text(json.dumps(res, indent=1))
    except OSError: pass

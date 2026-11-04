"""Score every */execution/trajectory.npz (or */evaluation/execution/trajectory.npz) under the given roots against compose (uscore)."""
import sys, glob, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import uscore as U
rows = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(f'{root}/**/execution/trajectory.npz', recursive=True)):
        try: o = U.score(U.load_traj(f))
        except Exception as e: print(f, 'ERR', e); continue
        r = max(o['beige_cup'][0] / 2.4, o['white_pot'][0] / 2.4, o['beige_cup'][1] / 24, o['white_pot'][1] / 24); rows.append((r, f, o))
for r, f, o in sorted(rows): print('%.3f  cup %5.2f/%5.1f  pot %5.2f/%5.1f  %s' % (r, *o['beige_cup'], *o['white_pot'], f.replace('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamU_C8/', '').replace('/execution/trajectory.npz', '')))

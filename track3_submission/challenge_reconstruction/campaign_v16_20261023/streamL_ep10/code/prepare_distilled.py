"""Copy one actor_grid actor (training/ + evaluation/) to a distilled dir and record its teacher in training_report.json
(same convention as refinement_20261021_ep10/distilled; policy.npz byte-identical). usage: prepare_distilled.py ACTOR_DIR DEST TEACHER_DIR"""
import sys, json, shutil
from pathlib import Path
src, dest, teacher = (Path(p).resolve() for p in sys.argv[1:4])
if dest.exists(): raise SystemExit('exists: ' + str(dest))
shutil.copytree(src, dest)
rp = dest / 'training/training_report.json'; r = json.loads(rp.read_text()); r['teacher'] = str(teacher); r['actor_grid_source'] = str(src)
rp.write_text(json.dumps(r, indent=2) + '\n'); print('prepared', dest)

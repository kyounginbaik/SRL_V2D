"""X8: copy a distilled actor dir to prepared/NAME with the teacher recorded in training_report.json (stream G / N8 convention).
usage: prepare.py ACTOR_DIR TEACHER_DIR OUT_DIR"""
import sys, json, shutil
from pathlib import Path
a, t, o = (Path(x).resolve() for x in sys.argv[1:4])
if o.exists(): raise SystemExit('exists')
shutil.copytree(a, o); p = o / 'training/training_report.json'; r = json.loads(p.read_text())
r['teacher'] = str(t); r['prepared_note'] = 'stream X8: copy of the distilled actor dir with the teacher recorded (stream G stage_G_component.py convention); source ' + str(a)
p.write_text(json.dumps(r, indent=2) + '\n'); print('prepared', o)

"""write BASE/actor_report.json from a distill attempts.json row (keys as the phase-6 staging actor reports).
usage: mk_report.py DISTILL_DIR ACTOR_NAME BASE TEMPLATE_REPORT SELECTION_NOTE"""
import json, sys, os, shutil
d, name, base, tmpl, note = sys.argv[1:6]
a = json.load(open(d + '/attempts.json')); a = a if isinstance(a, list) else a.get('attempts', a)
r = [x for x in a if x['checkpoint'].split('/')[-3] == name][0]; t = json.load(open(tmpl))
for k in t:
    if k not in r: r[k] = t[k]
r['training_report'] = os.path.join(os.path.dirname(r['checkpoint']), 'training_report.json'); r['selection'] = note; r['reference_proxy_mp_sr'] = 1.0 if r['passed'] else 0.0
assert os.path.exists(r['training_report']) and os.path.exists(r['checkpoint'])
os.makedirs(base, exist_ok=False); shutil.copy2('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/input_manifest.json', base)
json.dump(r, open(base + '/actor_report.json', 'w'), indent=2); print(json.dumps({k: r[k] for k in ('checkpoint', 'object_errors', 'passed')}))

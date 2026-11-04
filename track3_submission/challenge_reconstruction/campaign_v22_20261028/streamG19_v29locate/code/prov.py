"""Provenance of every build-v2 candidate: world input_dir(s) of its legacy component -> the world's *_reference.json / input_manifest
source dir -> class OUR (campaign_v22 target_final_v* / reconstruction_recovery selected20_final_*) | COMPOSE | OTHER."""
import json, glob, os, re, subprocess, sys
from pathlib import Path
CD = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'; OV = '/home/exaFLOPs/v22_overflow_20261028/common_v2_candidates'
out = {}
def classify(s):
    if re.search(r'compose|ComPose|final_recon_ours|cowork', s, re.I) and not re.search(r'target_final_v\d', s): return 'COMPOSE'
    if re.search(r'campaign_v22_20261028/target_final_v\d|selected20_final_v\d|reconstruction_recovery_20261026/selected20', s): return 'OUR'
    return 'OTHER'
for d in sorted(glob.glob(f'{CD}/*/episode_*') + glob.glob(f'{OV}/*/episode_*')):
    tag = d.split('/')[-2]; ep = int(d[-6:])
    try: c = json.load(open(f'{d}/completed.json'))
    except Exception: continue
    leg = c.get('legacy_component', '')
    found = subprocess.run(['grep', '-rhoE', '"input_dir": "[^"]*"', leg], capture_output=True, text=True).stdout
    dirs = sorted({l.split('"')[3] for l in found.splitlines() if 'portable_validation' not in l and 'refinement_' not in l})
    srcs = []
    for w in dirs:
        root = Path(w).parents[1]
        for j in sorted(root.glob('*_reference.json')) + sorted(root.glob('*reference*.json')):
            try:
                r = json.load(open(j)); srcs += [str(v) for k, v in r.items() if k.endswith('_dir') and isinstance(v, str)]
            except Exception: pass
        im = Path(w) / 'input_manifest.json'
        if not srcs and im.exists():
            txt = im.read_text(); srcs += re.findall(r'/data1/[^"]*(?:target_final_v\d+\w*|compose\w*|final_recon\w*|selected20_final_\w+)', txt)[:3]
    cls = sorted({classify(s) for s in srcs}) or ['NOSRC']
    out[f'{ep}/{tag}'] = dict(ep=ep, tag=tag, worlds=dirs, sources=sorted(set(srcs)), cls=cls,
                             exact=c.get('exact_previous_physical_states_actions') is True)
    print(ep, tag, cls, sorted(set(srcs))[:2], flush=True)
json.dump(out, open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/out/provenance.json', 'w'), indent=1)

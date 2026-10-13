"""Summarize completed frozen-policy episode-delta probes without loading arrays."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
a=p.parse_args()
rows=[]
for f in sorted(a.root.glob('*/seed_*/summary.json')):
    r=json.loads(f.read_text())
    rows.append({k:r[k] for k in ('variant','seed','dimensions','accepted','gate_pass_count',
                                 'improving_count','baseline_cost','best_raw_cost','best_valid_cost','seconds')})
print(json.dumps(rows,indent=2,allow_nan=False))

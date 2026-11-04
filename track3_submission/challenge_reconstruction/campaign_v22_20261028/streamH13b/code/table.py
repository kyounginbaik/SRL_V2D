"""H13b results table: score_any vs target_final_v12b_g15 of the v27 baseline (G15) and every H13 tag. usage: table.py"""
import json
from pathlib import Path
S = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/scores/cand')
for ep in (19, 3, 8, 1, 14, 25, 30, 16, 10, 15):
    b = next(S.glob(f'G15_v27_ep{ep:02d}_vs_v12bg15'), None) or S / f'G15_v27_ep{ep}_vs_v12bg15'; rows = [(f'v27 ({b.name})', b / 'score.json')] + [(p.name, p / 'score.json') for p in sorted(S.glob(f'H13_ep{ep:02d}_*'))]
    for name, f in rows:
        if not f.exists(): continue
        d = json.loads(f.read_text())[str(ep)]; s = d['scores']; o = ', '.join(f'{k} {v[0]:.2f}/{v[1]:.1f}' for k, v in d['objects'].items())
        print(f"| {ep} | {name} | {o} | {s['MPPE']:.3f} | {s['AUC']:.4f} | {int(s['MP-SR'])} {int(s['SP-SR'])} |")

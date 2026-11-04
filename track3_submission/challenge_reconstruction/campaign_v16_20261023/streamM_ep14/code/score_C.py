#!/usr/bin/env python3
"""Score one verified episode-14 rollout against compose_target with the unchanged score_vs_compose.py.
Builds a 1-episode submission parquet: the v21 package's episode-14 rows (row ids, provenance columns and object_mesh_b64 = our
packed meshes 090b7d17 rack / 78bd065c cup; the component must ship the same meshes, checked) with pos/quat replaced by the
achieved poses of RESULTS_PARQUET (the packer copies achieved_* verbatim: verified on v21, max diff 0).
usage: score_C.py RESULTS_PARQUET COMPONENT_DIR OUT_DIR"""
import sys, subprocess, hashlib, shutil, json
from pathlib import Path
import pandas as pd
V21 = Path('/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post')
SHIP = {'blue_cup.obj': '78bd065c74039ff91a21ad6532924c4f4797025228eeb127225a20e78f1ec6ec', 'plastic_dish_rack.obj': '090b7d17790ef35fb1c1364488cdec5adc7232f208dfb430494b752b1ab69289'}
res, comp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=False)
for f, h in SHIP.items():
    for d in (comp / 'meshes/episode_000014', V21 / 'meshes/episode_000014'):
        assert hashlib.sha256((d / f).read_bytes()).hexdigest() == h, (d, f)
s = pd.read_parquet(V21 / 'track3_submission.parquet'); s = s[s.row_id.str.contains('/e000014/')].copy()
r = pd.read_parquet(res)
for i, row in s.iterrows():
    f = int(row.row_id.split('/f')[1][:6]); o = int(row.row_id.split('/o')[1]); a = r[(r.step == f) & (r.object_id == o)].iloc[0]
    for k, c in zip(['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w'], ['achieved_position_x', 'achieved_position_y', 'achieved_position_z', 'achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']):
        s.at[i, k] = float(a[c])
s.to_parquet(out / 'submission_ep14.parquet'); (out / 'results').mkdir(); shutil.copy2(res, out / 'results/episode_000014.parquet')
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target'
subprocess.run(['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python', f'{C}/score_vs_compose.py', str(out / 'submission_ep14.parquet'), str(out / 'results'), C, str(out / 'score.json'), '14'], check=True)
print(json.dumps(json.load(open(out / 'score.json'))))

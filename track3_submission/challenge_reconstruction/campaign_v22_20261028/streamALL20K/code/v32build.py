"""ALL20-K v32 assembler. Base = v30c1_cwmesh_nogt parquet (== v29 rollouts + 17 GT-free coworker-mesh swaps aligned to our v29
meshes). For each episode in SPEC, take the pose + checkpoint columns from a donor package (a build-v2 rollout component, nothing
edited). Mesh payload: keep the base (cwmesh) payload when the donor's payload equals v29's (same body frame -> cwmesh alignment still
valid); otherwise take the donor's own payload. results/ = per-episode packer inputs from the donor (else v29).
usage: v32build.py SPEC.json OUT_PKG_DIR(.../all_20_episodes) LABEL
SPEC = {"33": "/abs/donor/all_20_episodes", ...}"""
import sys, json, shutil, hashlib, pandas as pd
from pathlib import Path
R = Path('/data1/SRL_V2D/challenge_submissions_SRL')
BASE = R / 'track3_v30c1_cwmesh_nogt_20261101/all_20_episodes'; V29 = R / 'track3_v29_20261030/all_20_episodes'
spec = {int(k): Path(v) for k, v in json.loads(Path(sys.argv[1]).read_text()).items()}
OUT, LAB = Path(sys.argv[2]), sys.argv[3]
assert not OUT.exists(), OUT
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
b = pd.read_parquet(BASE / 'track3_submission.parquet'); v29 = pd.read_parquet(V29 / 'track3_submission.parquet').set_index('row_id')
uni = ['eval_script_sha256', 'eval_code_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
POSE = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'checkpoint_sha256']
s = b.copy().set_index('row_id'); prov = {}
for ep, dp in sorted(spec.items()):
    d = pd.read_parquet(dp / 'track3_submission.parquet').set_index('row_id')
    assert set(d.index) == set(s.index) and all(set(d[c]) == set(s[c]) for c in uni), dp
    m = s.index.str.startswith(f'track_3/e{ep:06d}/')
    s.loc[m, POSE] = d.loc[s.index[m], POSE].to_numpy()
    mesh_rows = s.index[m & s.object_mesh_b64.fillna('').str.strip().ne('')]
    mesh_src = {}
    for r in mesh_rows:
        if d.at[r, 'object_mesh_b64'] == v29.at[r, 'object_mesh_b64']: mesh_src[r] = 'base(cwmesh or v29)'
        else: s.at[r, 'object_mesh_b64'] = d.at[r, 'object_mesh_b64']; mesh_src[r] = 'donor'
    prov[ep] = dict(donor=str(dp), donor_sha=h(dp / 'track3_submission.parquet'), checkpoint=d.loc[s.index[m], 'checkpoint_sha256'].iloc[0],
                    rows=int(m.sum()), mesh=mesh_src)
s = s.reset_index()[b.columns]
OUT.mkdir(parents=True)
s.to_parquet(OUT / 'track3_submission.parquet', index=False)
shutil.copytree(BASE / 'submission_kit', OUT / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.receipt.json'))
(OUT / 'results').mkdir()
for p in sorted((V29 / 'results').glob('episode_*.parquet')):
    ep = int(p.name[8:14]); src = (spec[ep] / 'results' / p.name) if ep in spec else p
    shutil.copy2(src, OUT / 'results' / p.name)
info = dict(label=LAB, base=str(BASE), base_sha=h(BASE / 'track3_submission.parquet'), sha=h(OUT / 'track3_submission.parquet'),
            episodes_replaced=prov)
(OUT / 'V32_BUILD.json').write_text(json.dumps(info, indent=1))
print(LAB, info['sha'][:8], {e: (p['rows'], p['checkpoint'][:8]) for e, p in prov.items()})

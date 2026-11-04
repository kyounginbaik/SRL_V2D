"""Z9 command list: eval (final_v5) + public/held-out (r8_base) x {evidence, consensus} and 6-deg injection tests."""
import json
from pathlib import Path
import pyarrow.parquet as pq
W = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
Z = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon')
PY = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B ' + str(Z / 'code/z9_anchor.py')
roster_eval = json.load(open('/data1/SRL_V2D/challenge_submissions_SRL/track3_v22b_20261028/all_20_episodes/submission_kit/data/track_3_evaluation_objects.json'))['episodes']
def pub_roster(ep):
    t = pq.read_table(f'/data1/video_to_data_challenge/track_3/public/data/chunk-000/episode_{ep:06d}.parquet', columns=['observation.objects']).to_pandas()
    return [o['name'] for o in t['observation.objects'].iloc[0]]
lines = []
for ep, names in sorted(roster_eval.items(), key=lambda x: int(x[0])):
    ep = int(ep); root = W / 'canonical/evaluation' / f'episode_{ep:06d}'
    for mode, tag in [('final', 'z9_anchor_final')]:
        lines.append(f"{PY} {root} {W}/selected20_final_v5/episode_{ep:06d} --roster {','.join(names)} --mode {mode} --report {Z}/out/eval{ep:02d}_{tag}.json --out {root}/exports/{tag}")
for split in ['public', 'public_heldout']:
    for root in sorted((W / 'canonical' / split).glob('episode_*')):
        ep = int(root.name[-6:]); r = ','.join(pub_roster(ep)); base = root / 'exports/r8_base'
        for mode, tag, inj in [('final', 'z9_anchor_final', 0), ('final', 'z9_inj6_final', 6)]:
            lines.append(f"{PY} {root} {base} --roster {r} --mode {mode} --inject {inj} --report {Z}/out/pub{ep:02d}_{tag}.json --out {root}/exports/{tag}")
(Z / 'code/cmds_final.txt').write_text('\n'.join(lines) + '\n'); print(len(lines))

W=/home/hansol/v2d/outputs/variants_20261030/CD_work
CAN=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical
/home/hansol/v2d/.venvs/v2d-host/bin/python - <<'EOF'
import json, glob, os
W = '/home/hansol/v2d/outputs/variants_20261030/CD_work'; CAN = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical'
for f in sorted(glob.glob(f'{W}/pub/*/episode_*/exports/*/public_score.json')):
    rel = os.path.relpath(f, f'{W}/pub'); mine = json.load(open(f)); ref = json.load(open(f'{CAN}/{rel}'))
    same = mine['public_errors'] == ref['public_errors'] and mine['public_scores'] == ref['public_scores']
    print(rel, 'IDENTICAL to stored' if same else 'DIFFERS', {k: round(v, 4) for k, v in mine['public_scores'].items()}, {o: {k: round(v, 3) for k, v in e.items()} for o, e in mine['public_errors'].items()})
EOF

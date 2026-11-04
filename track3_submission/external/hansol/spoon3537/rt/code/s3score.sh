#!/bin/bash
# spoon3537: proxy score of a build-v2 candidate vs a target with the team scorer (make_sub_any + score_vs_compose = kit Track 3 functions)
# usage: s3score.sh EP CAND_TAG LABEL TARGET_DIR -> /data1/hansol/spoon3537/rt/scores/LABEL/score.json + one summary line
set -euo pipefail
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
E=$1; NM=$(printf episode_%06d $E); D=/data1/hansol/spoon3537/rt/candidates/$2/$NM; case $2 in /*) D=$2/$NM;; esac; O=/data1/hansol/spoon3537/rt/scores/$3; T=$4
rm -rf $O; cd /tmp && OMP_NUM_THREADS=1 $PY -B $C/code/make_sub_any.py $E $D $O >/dev/null && OMP_NUM_THREADS=1 $PY -B /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py $O/submission.parquet $O/results $T $O/score.json $E 2>&1 | grep -v Warn > $O/score.log || true
python3 -c "import json;d=json.load(open('$O/score.json'))['$E'];o=d['objects'];s=d['scores'];print('SCORE $3 vs $T', ' '.join(f'{k} {v[0]:.3f}cm/{v[1]:.2f}deg' for k,v in o.items()), f\"MPPE {s['MPPE']:.3f} AUC {s['AUC']:.4f} MP-SR {s['MP-SR']} SP-SR {s.get('SP-SR')}\")"

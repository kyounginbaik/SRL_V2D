#!/bin/bash
# usage: score_candidate.sh CANDIDATE_EPISODE_DIR LABEL
# Packs a submission whose ep25 rows come from the candidate (results/episode_000025.parquet + meshes/episode_000025) with the unchanged kit packer
# (v21 package kit), then scores it against compose_target with score_vs_compose.py (official anchor frame-0 procedure).
set -euo pipefail
C=$(realpath $1); L=$2
U=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25
KIT=/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post/submission_kit
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
V21=/data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes_post
O=$U/scoresC/$L; mkdir -p $O/results $O/meshes
# the packer requires all 20 episodes: v21 rows for the other 19, the candidate for ep25 (only ep25 is scored)
for f in $V21/results/episode_*.parquet; do [ "$(basename $f)" = episode_000025.parquet ] || cp $f $O/results/; done
for d in $V21/meshes/episode_*; do [ "$(basename $d)" = episode_000025 ] || cp -r $d $O/meshes/; done
cp $C/results/episode_000025.parquet $O/results/; cp -r $C/meshes/episode_000025 $O/meshes/
URL=$($PY -c "import json;print(json.load(open('$C/build_manifest.json'))['code_commit_url'])")
(cd $KIT && $PY tools/pack_submission.py --track track_3 --results $O/results --meshes $O/meshes --code-commit $URL -o $O/sub.parquet) > $O/pack.log 2>&1 || { tail -20 $O/pack.log; exit 1; }
cd /tmp && $PY /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py $O/sub.parquet $O/results /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target $O/score.json 25

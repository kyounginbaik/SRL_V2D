#!/bin/bash
R=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py
P=/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes
O=$R/out/regsens; mkdir -p $O
for lab in base long5 long10 mid5 mid10 short5 short10; do
  T=$R/tmp_regsens/$lab; [ $lab = base ] && T=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17
  for ep in 33 36 5 3 6 37; do echo "$PY -B -u $S $P/track3_submission.parquet $P/results $T $O/${lab}_ep$ep.json $ep > $O/${lab}_ep$ep.log 2>&1"; done
done | xargs -P 8 -I{} sh -c "{}"
cd $O; for lab in base long5 long10 mid5 mid10 short5 short10; do echo -n "$lab "; for ep in 33 36 5 3 6 37; do python3 -c "
import json;d=json.load(open('${lab}_ep$ep.json'))['$ep'];o=d['objects']['wooden_spoon'];print('ep$ep %.2f/%.1f'%(o[0],o[1]),end=' ')"; done; echo; done

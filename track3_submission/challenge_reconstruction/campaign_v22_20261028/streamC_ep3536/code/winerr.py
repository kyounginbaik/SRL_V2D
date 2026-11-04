"""per-window pos/rot errors of plan rollouts (scorer-identical). usage: winerr.py BASE JSON_LIST"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cemC
w = cemC.Worker(sys.argv[1])
for p in json.loads(sys.argv[2]):
    q = dict(cemC.plan36.DEFAULT); q.update(p); Q, _ = w.rollout(q); pe, re = w.score(Q)
    print(json.dumps(p)); print('  mean', round(pe.mean(), 2), round(re.mean(), 1)); print('  pos', [round(float(pe[i:i + 20].mean()), 1) for i in range(0, len(pe), 20)]); print('  rot', [round(float(re[i:i + 20].mean())) for i in range(0, len(re), 20)])

"""Find, for each v29 episode component, the saved actor dir whose policy phase_weights equal the shipped component's. usage: findactor.py OUT.json"""
import json, glob, sys, numpy as np
C22='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'; CV='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
sel=json.load(open(f'{C22}/selection_v29.json')); out={}
roots=['/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028','/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023','/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026']
for e,tag in sel.items():
    ep=int(e); w=np.load(f'{CV}/{tag}/episode_{ep:06d}/backend/policy_relocated.npz',allow_pickle=False)['phase_weights']
    hits=[]
    for r in roots:
        for p in glob.glob(f'{r}/*/runs/save_*ep{ep:02d}*/actor/training/policy.npz')+glob.glob(f'{r}/*/runs/save_*ep{ep}_*/actor/training/policy.npz'):
            try:
                w2=np.load(p,allow_pickle=False)['phase_weights']
                if w2.shape==w.shape and np.array_equal(w2,w): hits.append(p.rsplit('/training',1)[0])
            except Exception: pass
    out[e]=dict(tag=tag,actors=sorted(set(hits))); print(e,tag,sorted(set(hits))[:3],flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)

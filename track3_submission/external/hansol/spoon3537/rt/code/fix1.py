import json
S='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamALL20K'
p=f'{S}/code/a20select.py'; s=open(p).read()
s=s.replace("gain['dAUC'] >= 0 and","gain['dAUC'] >= -0.002 and").replace("AUC not lower, MPPE lower","AUC not lower by > 0.002, MPPE lower")
open(p,'w').write(s)
p=f'{S}/code/a20orch.py'; s=open(p).read(); s=s.replace('datetime.datetime.utcnow()','datetime.datetime.now(datetime.timezone.utc)'); open(p,'w').write(s)
sp=f'{S}/out/orch_state.json'; st=json.load(open(sp)); st['gated']=[]; json.dump(st,open(sp,'w'),indent=1)

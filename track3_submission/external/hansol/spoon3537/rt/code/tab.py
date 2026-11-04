import json,sys
P,E=sys.argv[1],sys.argv[2]
v14=json.load(open(f'{P}/target_final_v14_g17_scores.json')); out={}
for e in [1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]:
    row=[]; out[e]={}
    for lab,d in (('v14',v14[str(e)]),)+tuple((v,json.load(open(f'{E}/out/v29_vs_E14_{v}_n100/ep{e}.json'))[str(e)]) for v in ('HGA','HGB')):
        o=' '.join(f"{k[:6]} {x[0]:.2f}/{x[1]:.1f}" for k,x in d['objects'].items()); s=d['scores']; out[e][lab]=d
        row.append(f"{lab}: {o} MP{s['MP-SR']:.0f} MPPE {s['MPPE']:.2f} AUC {s['AUC']:.3f}")
    print(e,' | '.join(row))
json.dump(out,open(f'{E}/out/v29_scores_3targets.json','w'),indent=1)

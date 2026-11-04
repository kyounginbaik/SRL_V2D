import json,sys
n,rn=sys.argv[1:]
try:
    for u in json.load(open(f'/scratch2/exaFLOPs/srl_yam_run/comparison/results/upper_test_{rn}_s42.json')):print(u['steps']//10**6,u['kinds'],round(u['passed'],3),round(u['cm'],2),round(u['deg'],1),'rise',round(u['object_max_rise_cm'],1),[round(h['passed'],2) for h in u.get('holds',[])])
except Exception as e:print('no score',e)

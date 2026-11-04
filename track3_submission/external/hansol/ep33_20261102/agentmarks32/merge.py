import json
from collections import Counter
raw={}
for t in 'ABC': raw.update(json.load(open(f'agentmarks32_{t}.json')))
ann={str(int(k)):(v['mark'] if isinstance(v['mark'],str) else [float(x) for x in v['mark']]) for k,v in raw.items() if v['mark']!='unsure'}
json.dump(ann,open('agent_marks.json','w'),indent=0)
u=json.load(open('marks_ep32.json'))
cls=lambda m:'none' if m is None else (m if isinstance(m,str) else 'edge')
rows=[(int(k),u[k],raw.get(k,{}).get('mark'),raw.get(k,{}).get('conf')) for k in sorted(u,key=int)]
for r in rows: print(r)
print(Counter((cls(r[1]),cls(r[2])) for r in rows)); print('agent kept',len(ann),'of',len(raw))

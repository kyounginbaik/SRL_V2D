import json
raw={}
for t in 'ABC': raw.update(json.load(open(f'agentmarks_{t}.json')))
ann={}
for k,v in sorted(raw.items(),key=lambda x:int(x[0])):
    m=v['mark']
    if m=='unsure': continue
    ann[str(int(k))]=m if isinstance(m,str) else [float(m[0]),float(m[1])]
json.dump(ann,open('agent_marks.json','w'),indent=0)
u=json.load(open('../orient/ep33_marks_merged_20261103d.json')); u.pop('_comment',None)
import math
def cls(m): return m if isinstance(m,str) else 'edge'
agree=dis=0; rows=[]
for k in sorted(u,key=int):
    a=raw.get(str(int(k)))
    am=a['mark'] if a else None
    um=u[k]
    rows.append((int(k),um if isinstance(um,str) else [round(x,2) for x in um[:2]],am,a['conf'] if a else None))
for r in rows: print(r)
from collections import Counter
print(Counter((cls(r[1]), cls(r[2]) if r[2] is not None else 'none') for r in rows))
print('agent marks kept', len(ann))

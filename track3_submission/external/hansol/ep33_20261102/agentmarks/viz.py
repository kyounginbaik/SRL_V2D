import re,json
exec(open('summ.py').read().split('for who')[0])
raw={}
for t in 'ABC': raw.update(json.load(open(f'agentmarks_{t}.json')))
u=json.load(open('../orient/ep33_marks_merged_20261103d.json')); u.pop('_comment',None)
R={(w,k):load(f'rep_{w}_{k}.txt') for w in ('agent','user') for k in ('r6','h5')}
def enc(m):
    if m is None: return None
    if isinstance(m,str): return m
    return [round(m[0],2),round(m[1],2)]
rows=[]
for f in range(0,347,2):
    a=raw.get(str(f),{}); um=u.get(str(f))
    r6=R[('agent','r6')].get(f) or R[('user','r6')].get(f); h5=R[('agent','h5')].get(f) or R[('user','h5')].get(f)
    rows.append(dict(f=f,h=enc(um),a=enc(a.get('mark')),c=a.get('conf'),w=a.get('why'),
      r6=r6[1] if r6 else None,h5=h5[1] if h5 else None,
      ar6=R[('agent','r6')][f][3] if f in R[('agent','r6')] else None, ah5=R[('agent','h5')][f][3] if f in R[('agent','h5')] else None,
      ur6=R[('user','r6')][f][3] if f in R[('user','r6')] else None, uh5=R[('user','h5')][f][3] if f in R[('user','h5')] else None))
print(json.dumps(rows,separators=(',',':')))

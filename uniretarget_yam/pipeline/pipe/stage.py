"""Print what one stage of an episode's contact schedule searches, and write the join description it needs."""
import json,sys
meta,i,parquet,out=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
CARRY_ALONE=20
Mt=json.load(open(meta));H=Mt['holds'];h=H[i];earlier=H[:i];own=[x for x in earlier if x['side']==h['side']]
same=[x for x in earlier if x['index']==h['index']]
later=sorted({x['index'] for x in H if x['index']!=h['index'] and not any(y['index']==x['index'] for y in H[:i])})
together=[x for x in same if x.get('release') is None]
mid=bool(h.get('join') or own or together)
rel=[dict(side=x['side'],frame=int(x['release'])) for x in earlier if x.get('release') is not None and x['side']!=h['side'] and x['index']==h['index']]
end=int(h.get('release') or h.get('object_end') or h['end'])
# Two hands carrying together: this hand only has to carry alone until the next one is on the object. Its tape still runs to the end, so the next stage searches with both.
joins=[x for x in H[i+1:] if x['index']==h['index']]
if joins and h.get('release') is None:end=min(end,max(int(joins[0]['onset']),int(h['onset']))+CARRY_ALONE)
d=dict(start=(int(h['onset']) if together else int(h['start'])) if mid else None,parquet=parquet or None,release=rel)
if own:d['from']=int(own[-1].get('release') or own[-1]['end'])+8
if i==0:d=None
if d is not None:json.dump(d,open(out,'w'))
names=[o['name'] for o in Mt['objects']];print(h['index'],h['side'],end,','.join(map(str,later)) or '-',int(mid),len(H),','.join(str(k) for k,n in enumerate(names) if n in Mt.get('frozen',[])) or '-')

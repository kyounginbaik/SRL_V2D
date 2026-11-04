import re,json
def load(p):
    d={}
    for l in open(p):
        m=re.match(r"(\d+) (.+?) \{'facing': (-?[\d.]+), 'arrow_err': ([\d.]+|None), 'ok': (\w+)",l)
        if m: d[int(m[1])]=(m[2],float(m[3]),None if m[4]=='None' else float(m[4]),m[5]=='True')
    return d
for who in ('agent','user'):
    for k in ('r6','c18','v14','h5'):
        d=load(f'rep_{who}_{k}.txt')
        if not d: print(who,k,'EMPTY'); continue
        cat=lambda s: 'edge' if s.startswith('(') or s.startswith('[') else s.strip("'")
        ok={c:[0,0] for c in ('out','in','edge')}
        for f,(a,fc,e,o) in d.items():
            c=cat(a); ok[c][0]+=o; ok[c][1]+=1
        print(f'{who:5s} {k:4s} ok {sum(v[0] for v in ok.values())}/{len(d)}', {c:f'{v[0]}/{v[1]}' for c,v in ok.items()})

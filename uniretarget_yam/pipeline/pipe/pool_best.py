"""Write the best grasp any stage found to pool.txt and flag it when the Newton gate did not pass. Usage: pool_best.py EP"""
import json,glob,os,sys
EP=sys.argv[1];S='/scratch2/exaFLOPs/srl_yam_run/search/'+EP;b=None
for f in glob.glob(S+'/stage*_s*/check.txt')+glob.glob(S+'/s*/check_*.txt'):
    r=[json.loads(l) for l in open(f) if l.startswith('{')]
    if not r:continue
    d=os.path.dirname(f);nm=os.path.basename(f);sub='flash_pass' if nm in('check.txt','check_pass.txt') else 'flash_'+nm[6:-4]
    if not os.path.isfile(f'{d}/{sub}/reference.parquet'):continue
    st=int(os.path.basename(d)[5]) if os.path.basename(d).startswith('stage') else 1
    k=(st,min(x.get('lift',0) for x in r)>=.8,min(sum(v for a,v in x['kinds'].items() if a in('held','off_track'))/sum(x['kinds'].values()) for x in r),min(x['frontier'] for x in r))
    if b is None or k>b[0]:b=(k,f"/search/{EP}/{os.path.basename(d)}/{sub}/reference.parquet")
open(S+'/pool.txt','w').write(b[1]+'\n' if b else '');open(S+'/partial.txt','w').write('PARTIAL_GATE\n');print(EP,b)

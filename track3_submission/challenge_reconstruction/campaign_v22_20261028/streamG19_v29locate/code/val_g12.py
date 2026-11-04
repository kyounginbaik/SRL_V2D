import json,os,glob,sys
S,G=sys.argv[1:3]
bad=0;n=0
for lab,g in (('scanv9','scanv9'),('c21','c21'),('c27','c27')):
  for f in glob.glob(f'{S}/out/tab/{lab}/ep*.json'):
    e=os.path.basename(f)[2:-5]; d=json.load(open(f))
    for p,v in d.items():
      if p=='v29' or 'mp' not in v: continue
      gf=f'{G}/out/pkg/{p}_{g}/ep{e}.json'
      if not os.path.exists(gf): continue
      s=json.load(open(gf))[e]; sc=s.get('scores') or {}
      if sc.get('MP-SR') is None: continue
      o=(int(sc['SP-SR']<1),int(sc['MP-SR']<1)); n+=1
      if o!=(v['sp'],v['mp']): bad+=1; print('DIFF',lab,e,p,o,(v['sp'],v['mp']),s['objects'],v['obj'])
print(n,'compared',bad,'diff')

import sys,json
for i,l in enumerate(x for x in open(sys.argv[1]) if x.startswith('{"objects')):
  a,b=l.split('} {',1); a=json.loads(a+'}'); b=json.loads('{'+b)
  o=a['objects']['wooden_spoon']; print(i, round(o[0],2), round(o[1],1), a['win'], {k:v for k,v in b.items() if k in ('gr','gp','sig') or (k.endswith('_a') and v)})

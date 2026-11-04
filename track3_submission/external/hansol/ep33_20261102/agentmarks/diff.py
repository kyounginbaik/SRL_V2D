import re,json
exec(open('summ.py').read().split('for who')[0])
raw={}
for t in 'ABC': raw.update(json.load(open(f'agentmarks_{t}.json')))
for who in ('agent','user'):
    a=load(f'rep_{who}_r6.txt'); b=load(f'rep_{who}_h5.txt')
    print('==',who,'frame mark conf | r6 facing ok | h5 facing ok')
    for f in sorted(a):
        if a[f][3]!=b[f][3]:
            cf=raw.get(str(f),{}).get('conf') if who=='agent' else ''
            print(f, a[f][0], cf, '| r6', a[f][1], a[f][2], a[f][3], '| h5', b[f][1], b[f][2], b[f][3])

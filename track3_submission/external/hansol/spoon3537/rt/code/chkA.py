import sys, hashlib, pandas as pd
R='/data1/hansol/spoon3537/rt/packages/'
pa=R+sys.argv[1]+'/all_20_episodes/track3_submission.parquet'
a=pd.read_parquet(pa).set_index('row_id')
b=pd.read_parquet('/data1/hansol/teacher33/packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105/all_20_episodes/track3_submission.parquet').set_index('row_id')
d=(a!=b)&~(a.isna()&b.isna()); r=d.any(axis=1); print('vs combined final: episodes changed', sorted(set(i.split('/')[1] for i in a.index[r])), 'rows', int(r.sum()))
for ep, pk in (('e000036','track3_FINALc_ep36s1d_tgt_s1'), ('e000037','track3_FINALc_ep37s1a_tgt_s1')):
    c=pd.read_parquet(R+pk+'/all_20_episodes/track3_submission.parquet').set_index('row_id'); m=a.index.str.contains('/'+ep+'/')
    dd=(a[m]!=c[m])&~(a[m].isna()&c[m].isna()); print(ep, 'rows identical to', pk, not dd.any().any())
print('sha', hashlib.sha256(open(pa,'rb').read()).hexdigest())

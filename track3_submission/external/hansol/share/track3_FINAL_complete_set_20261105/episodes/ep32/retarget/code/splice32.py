"""Replace only the ep32 pose rows of a base parquet with those of a donor parquet. Mesh cells and all other episodes untouched.
usage: splice32.py BASE.parquet DONOR.parquet OUT.parquet"""
import sys,hashlib,pandas as pd,numpy as np
b=pd.read_parquet(sys.argv[1]);d=pd.read_parquet(sys.argv[2]);assert list(b.columns)==list(d.columns),'columns differ'
POSE=[c for c in b.columns if c.startswith(('pos_','quat_','rot_')) or c in ('checkpoint_sha256',)];print('replace columns',POSE)
mb=b.row_id.str.contains('/e000032/');md=d.row_id.str.contains('/e000032/');assert mb.sum()==md.sum()==518,(mb.sum(),md.sum())
bi=b.index[mb];di=d.set_index('row_id').loc[b.loc[bi,'row_id']]
o=b.copy();o.loc[bi,POSE]=di[POSE].to_numpy()
ch=(o!=b)&~(o.isna()&b.isna());r=ch.any(axis=1);eps=sorted(set(i.split('/')[1] for i in o.row_id[r]));print('rows changed',int(r.sum()),'eps',eps,'cols',list(ch.columns[ch.any()]))
assert set(eps)<= {'e000032'}
o.to_parquet(sys.argv[3],index=False);print('sha',hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()[:16])

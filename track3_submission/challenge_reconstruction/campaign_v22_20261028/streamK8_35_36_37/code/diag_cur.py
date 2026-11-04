"""per-window errors of a build-v2 output vs target_final_v5 (scorer-identical, tlibK). usage: diag_cur.py EP TAGDIR_or_parquet"""
import sys, numpy as np; sys.path.insert(0, __file__.rsplit('/',1)[0]); import tlibK as T
ep = int(sys.argv[1]); p = sys.argv[2]
if not p.endswith('.parquet'): p = f'{p}/episode_{ep:06d}/results/episode_{ep:06d}.parquet'
Ta, _ = T.results_track(p); pe, re = T.errors(ep, Ta)
print('mean', round(pe.mean(), 3), round(re.mean(), 2), 'max', round(pe.max(), 2), round(re.max(), 1), 'n', len(pe))
for i in range(0, len(pe), 20): print(f'{i/20:5.1f}s pos {pe[i:i+20].mean():5.2f} (max {pe[i:i+20].max():5.2f})  rot {re[i:i+20].mean():5.1f} (max {re[i:i+20].max():5.1f})')

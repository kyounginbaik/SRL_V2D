"""usage: diag.py EP TAG_DIR [BIN_S=1.0] : per-time-bin scorer-aligned errors of achieved and of the physics reference vs target_final_v5"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import blib as B
ep, tag = int(sys.argv[1]), sys.argv[2]; bs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
reg = B.register(ep, B.comp_meshes(tag)); a, b = B.results_tracks(f'{tag}/results/episode_{ep:06d}.parquet')
ea = B.per_frame(ep, a, reg); eb = B.per_frame(ep, b, reg)
print('mean achieved', {n: (round(float(ea[n][0].mean()), 2), round(float(ea[n][1].mean()), 1)) for n in B.NAMES})
print('mean physref ', {n: (round(float(eb[n][0].mean()), 2), round(float(eb[n][1].mean()), 1)) for n in B.NAMES})
N = len(ea['blue_cup'][0]); k = int(round(bs * 20))
print('bin   | ach cup cm/deg | ach spoon cm/deg | ref cup | ref spoon')
for s in range(0, N, k):
    sl = slice(s, s + k); f = lambda e, n: f"{e[n][0][sl].mean():5.2f}/{e[n][1][sl].mean():5.1f}"
    print(f"{s/20:5.1f} | {f(ea,'blue_cup')} | {f(ea,'wooden_spoon')} | {f(eb,'blue_cup')} | {f(eb,'wooden_spoon')}")

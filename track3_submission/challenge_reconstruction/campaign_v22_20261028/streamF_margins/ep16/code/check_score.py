import sys, numpy as np; sys.path.insert(0, sys.path[0]); import tlib16 as L
C='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
for tag in sys.argv[1:]:
    Ta = L.results_tracks(f'{C}/{tag}/episode_000016/results/episode_000016.parquet'); e = L.errors(Ta)
    print(tag, {o: (round(float(p.mean()), 2), round(float(d.mean()), 1), round(float(p.max()), 1)) for o, (p, d) in e.items()})
    p, d = e['wooden_spoon']; print('  spoon per 20f cm', [round(float(p[i:i+20].mean()), 1) for i in range(0, len(p), 20)], 'deg', [round(float(d[i:i+20].mean()), 0) for i in range(0, len(d), 20)])

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
G = np.array(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json').read_text())['G'])
reg = registration(packed_meshes()); comp = compose_tracks()
for p in sys.argv[1:]:
    t = np.load(p); on = list(t['object_names']); ach = {n: to_T(t['object_positions'][:, on.index(n)], t['object_quaternions_wxyz'][:, on.index(n)]) for n in NAMES}
    if 'relabel' in sys.argv[0] or True: ach['mini_sweeper'] = ach['mini_sweeper'] @ G
    e = scorer_errors(ach, comp, reg); print(p, {n: (round(e[n]['cm'], 2), round(e[n]['deg'], 1)) for n in NAMES})
    for k in range(0, 235, 10): print('  %.1f' % (k / 20), ' '.join('%s %.1f/%2.0f' % (n[5:9], e[n]['cm_t'][k], e[n]['deg_t'][k]) for n in NAMES))

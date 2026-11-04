import sys, json, glob
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code')
from common19F import *
for d in sys.argv[1:]:
    for r in sorted(glob.glob(d + '/actor_*/evaluation/result.json')):
        j = json.load(open(r)); cs, _ = compose_score_traj(j['execution']['trajectory'])
        w = max(cs['mini_sweeper'][0] / 2.4, cs['mini_sweeper'][1] / 24, cs['mini_dust_pan'][0] / 2.4, cs['mini_dust_pan'][1] / 24)
        print('%.3f %s %s mp_sr=%s' % (w, r.split('/')[-3], cs, j['metrics']['reference_proxy_mp_sr']))

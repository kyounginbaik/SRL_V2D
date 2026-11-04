"""Lever sensitivity: our own target as truth; achieved = same track with a constant spoon body rotation (roll about the long
PCA axis, or tilt about the 2nd/3rd axis) of d deg. Kit-faithful time-mean errors per object. usage: sens.py TARGET_DIR"""
import sys, glob, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/code')
from k35lib import *
TG = sys.argv[1]; res = {}
for ep in (3, 5, 6, 32, 33, 35, 36, 37):
    d = f'{TG}/episode_{ep:06d}'; files = sorted(glob.glob(f'{d}/*_trajectory.npz'))
    # kit order = results object order; use v29 results
    import pandas as pd
    r = pd.read_parquet(f'/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/results/episode_{ep:06d}.parquet', columns=['object_id','object_body_name']).drop_duplicates().sort_values('object_id')
    names = list(r.object_body_name); tracks = [np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'] for n in names]
    infos = [mesh_info(f'{d}/{n}_mesh_metric.obj', n) for n in names]; s = names.index('wooden_spoon')
    line = {}
    for axn, ax in (('roll', 0), ('tiltA', 1), ('tiltB', 2)):
        for deg in (-15, -10, -5, 5, 10, 15):
            tr = list(tracks); tr[s] = body_rot(tracks[s], infos[s], infos[s]['pca'][:, ax] * np.radians(deg))
            p, o = aligned_errors(tr, tracks, infos)
            line[f'{axn}{deg:+d}'] = {n: [round(float(p[:, k].mean()), 2), round(float(o[:, k].mean()), 1)] for k, n in enumerate(names)}
    res[ep] = line
    print(ep, names, 'roll+-10:', line['roll+10'], line['roll-10'], '| tiltA+-10:', line['tiltA+10'], line['tiltA-10'], '| tiltB+-10:', line['tiltB+10'], line['tiltB-10'], flush=True)
json.dump(res, open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/out/lever_sens_v14.json', 'w'), indent=1)

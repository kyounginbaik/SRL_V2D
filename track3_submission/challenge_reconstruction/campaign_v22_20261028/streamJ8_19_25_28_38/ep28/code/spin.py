import sys, json, numpy as np
from scipy.spatial.transform import Rotation as R
run = sys.argv[1]; J = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28'
epd = J + '/world/episode_target/episode_000028'; f0 = int(sys.argv[2]) if len(sys.argv) > 2 else 140
r = json.load(open(run + '/result.json')); tr = np.load(r['execution']['trajectory']); nm = [str(x) for x in tr['object_names']]
ref = np.load(epd + '/reference_trajectory.npz'); i = nm.index('white_pot')
Ra = R.from_quat(tr['object_quaternions_wxyz'][:, i][:, [1, 2, 3, 0]]); Rr = R.from_quat(ref['object_quaternions_wxyz'][:, 1][:, [1, 2, 3, 0]])
ax = np.array([0, 1., 0])
for f in range(f0, len(Ra), 5):
    da = (Ra[f] * Ra[f0].inv()).as_rotvec(); dr = (Rr[f] * Rr[f0].inv()).as_rotvec()
    e = (Rr[f].inv() * Ra[f]).as_rotvec()   # body-frame error
    print(f, 'ach', np.round(np.degrees(da), 1), 'ref', np.round(np.degrees(dr), 1), 'err(body)', np.round(np.degrees(e), 1), 'spin about body y', round(float(np.degrees(e @ ax)), 1))

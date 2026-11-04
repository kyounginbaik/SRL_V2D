"""shift-only placement (psi = 0) of the settled standing twin; same objective as yaw_fit.py; saves twin_shift.npz (settled after 2 s)."""
import numpy as np
from scipy.optimize import minimize
exec(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code/yaw_fit.py').read().split("print('psi=0 no shift'")[0])
r = minimize(lambda x: err([0, *x])[0], [0, 0], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-6)); x = [0, *r.x]
for f in (1.0, 0.5):
    xx = [0, r.x[0] * f, r.x[1] * f]; S = pose(xx); S2, dp, dr = settle(S); print('frac', f, 'shift cm', np.round(np.array(xx[1:]) * 100, 2), 'err', np.round(err(xx), 2), 'settle drift %.3f cm %.2f deg' % (dp, dr))
    np.savez('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/twin_shift%s.npz' % ('' if f == 1 else '_half'), placed=S, settled=S2, x=np.array(xx))

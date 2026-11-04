"""J8 ep19: sensitivity scan - random knot perturbations (kcem19 KExp) at several scales in a window; prints cost distribution."""
import sys, json, numpy as np
sys.argv = sys.argv; sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code')
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import kcem19 as K
base, sides, t0, t1, knots = sys.argv[1], sys.argv[2].split(','), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
scales = [float(x) for x in sys.argv[6].split(',')]; n = int(sys.argv[7]); wk = int(sys.argv[8]) if len(sys.argv) > 8 else 4
EPD = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/world_st/episode_target/episode_000019'
r = json.load(open(base + '/result.json')); epd = r.get('input_dir', EPD); man = json.load(open(epd + '/input_manifest.json'))
spec = dict(scene=r['execution']['scene'], reference=base + '/reference.npz', original_reference=epd + '/tracking_reference.npz', source=man['source'], geometry=epd + '/metric_geometry.json',
            sides=sides, knots=knots, t0=t0, t1=t1, P={'mini_sweeper': .012, 'mini_dust_pan': .012}, D=12., peak_w=.1, pos_scale=.008, rot_scale=.05, sq_scale=.05)
if __name__ == '__main__':
    e = K.KExp(spec); rng = np.random.default_rng(0); b = e.evaluate(np.zeros(e.dim)); print('base', b['cost'], b['mean_position_cm'], flush=True)
    with ProcessPoolExecutor(wk, mp_context=mp.get_context('spawn'), initializer=K.init, initargs=(spec,)) as pool:
        for s in scales:
            X = [rng.normal(size=e.dim) * e.scale * s for _ in range(n)]; R_ = list(pool.map(K.ev, X))
            c = np.array([x['cost'] for x in R_]); p = np.array([x.get('mean_position_cm', [99, 99]) for x in R_])
            print('scale', s, 'cost med %.3f min %.3f  frac<base %.2f' % (np.median(c), c.min(), (c < b['cost']).mean()), 'pos med', np.median(p, 0).round(2), 'best', p[c.argmin()].round(2), flush=True)

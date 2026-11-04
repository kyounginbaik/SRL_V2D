"""per-0.5s scorer-faithful (evaluate_mp_sr proxy) error timeline of an execution trajectory.npz. usage: tl_traj.py EPDIR TRAJ [TRAJ2...]"""
import sys, json, numpy as np
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code']
import evaluate_mp_sr as EM
epd = sys.argv[1]; man = json.loads(open(epd + '/input_manifest.json').read())
cols = []
for tp in sys.argv[2:]:
    r = EM.evaluate_files(man['source'], tp, epd + '/input_manifest.json', None, epd + '/metric_geometry.json')
    ref = EM.load_reference(man['source']); tr = EM.resample_to_reference(EM.load_trajectory(tp), ref)
    geo = json.loads(open(epd + '/metric_geometry.json').read()); an, ax, ki = EM.validate_geometry(ref['object_names'], geo)
    for d in (ref, tr): d['object_positions'] = d['object_positions'] + np.einsum('twoij,oj->twoi', EM.rotation_matrices(d['object_quaternions_wxyz']), an)
    al, _ = EM.align_initial_scene(tr, ref, ax, ki)
    p = np.linalg.norm(al['object_positions'] - ref['object_positions'], axis=-1)[:, 0] * 100; a = np.degrees(EM.angular_errors(al['object_quaternions_wxyz'], ref['object_quaternions_wxyz'], ax, ki))[:, 0]
    print(tp, ' '.join('%.2f/%.1f' % (p[:, i].mean(), a[:, i].mean()) for i in range(p.shape[1]))); cols.append((p, a))
T = len(cols[0][0])
for s0 in range(0, T, 10): print(f'{s0/20:5.1f}s', ' | '.join(' '.join(f"{p[s0:s0+10, i].mean():5.2f}/{a[s0:s0+10, i].mean():5.1f}" for i in range(p.shape[1])) for p, a in cols))

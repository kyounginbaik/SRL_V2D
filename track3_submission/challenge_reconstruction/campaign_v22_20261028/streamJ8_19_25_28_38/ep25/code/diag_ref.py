import sys, json, numpy as np
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code']
import evaluate_mp_sr as EM
epd = sys.argv[1]; tp = sys.argv[2]; man = json.loads(open(epd + '/input_manifest.json').read())
ref = EM.load_reference(man['source']); tr = EM.resample_to_reference(EM.load_trajectory(tp), ref)
Rr = EM.rotation_matrices(ref['object_quaternions_wxyz'])[:, 0]; Ra = EM.rotation_matrices(tr['object_quaternions_wxyz'])[:, 0]
pr = ref['object_positions'][:, 0]; pa = tr['object_positions'][:, 0]
for s0 in range(0, len(pr), 10):
    row = []
    for i in range(2):
        v = np.linalg.norm(pr[min(s0+10, len(pr)-1), i] - pr[s0, i]) * 100
        # error rotation in body frame of ref: Rr^T Ra
        E = R.from_matrix(Rr[s0+5 if s0+5 < len(pr) else s0, i].T @ Ra[s0+5 if s0+5 < len(pr) else s0, i]).as_rotvec(); 
        dz = (pa[s0, i, 2] - pr[s0, i, 2]) * 100
        row.append(f'ref_move {v:4.1f}cm z {pr[s0,i,2]*100:5.1f} dz {dz:5.1f} errvec_body(deg) {np.degrees(E).round(0)}')
    # relative pose board2 in board1
    Trel_r = Rr[s0, 0].T @ (pr[s0, 1] - pr[s0, 0]); Trel_a = Ra[s0, 0].T @ (pa[s0, 1] - pa[s0, 0])
    print(f'{s0/20:5.1f}', ' || '.join(row), ' rel2in1 ref', (Trel_r*100).round(1), 'ach', (Trel_a*100).round(1))

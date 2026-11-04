"""Stream P ep33: re-score executed trajectories (unchanged evaluate_mp_sr.evaluate_files, declared source_from_setup transform,
ep33 metric_geometry) against one or more reference object_poses.npz files. usage: rescore.py TRAJ.npz POSES.npz [POSES2.npz ...]"""
import sys, json
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import evaluate_files
import os
EPD = os.environ.get('RESCORE_EPD', '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033')
import tempfile, os
def score(traj, poses):
    td = tempfile.mkdtemp(prefix='rescore_', dir=os.environ.get('TMPDIR', '/tmp')); os.symlink(os.path.abspath(poses), td + '/object_poses.npz'); os.symlink('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033/manifest.json', td + '/manifest.json'); poses = td
    m = evaluate_files(poses, traj, EPD + '/input_manifest.json', None, EPD + '/metric_geometry.json')['official_style_reference_proxy']
    return dict(mp_sr=m['mp_sr'], **{k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 2)) for k, v in m['objects'].items()})
if __name__ == '__main__':
    for p in sys.argv[2:]: print(p, json.dumps(score(sys.argv[1], p)), flush=True)

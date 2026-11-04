"""Score executed trajectories against the compose-retargeted ep35 reference (and the original v21 reference).
usage: rescore.py TRAJ_OR_DIR [...]   (a dir: uses DIR/execution/trajectory.npz or DIR/evaluation/execution/trajectory.npz)"""
import sys, json
from pathlib import Path
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import evaluate_files
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C')
NEW = (K / 'source/episode_000035', K / 'episode/episode_000035')
OLD = (Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000035'), Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter'))
for a in sys.argv[1:]:
    p = Path(a)
    if p.is_dir(): p = next(x for x in (p / 'execution/trajectory.npz', p / 'evaluation/execution/trajectory.npz') if x.exists())
    out = []
    for tag, (src, ed) in (('compose', NEW), ('v21ref', OLD)):
        m = evaluate_files(src, p, ed / 'input_manifest.json', None, ed / 'metric_geometry.json'); o = m['official_style_reference_proxy']['objects']['wooden_spoon']
        out.append(f"{tag} {o['mean_position_error_cm'][0]:.2f}/{o['mean_orientation_error_degrees'][0]:.1f} (max {o['max_position_error_m'][0]*100:.1f}/{o['max_orientation_error_degrees'][0]:.0f})")
    print(a.replace(str(K.parent), '.'), ' | '.join(out), flush=True)

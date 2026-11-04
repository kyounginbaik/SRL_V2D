import json, glob, numpy as np
from scipy.spatial.transform import Rotation as R
def ang(T): return float(np.degrees(R.from_matrix(np.array(T)[:3, :3]).magnitude()))
def expl(modes, tol=3.0):
    m = sum(1 for x in modes if x['d_mesh_mm'] < tol and x['d_twin_mm'] >= tol); t = sum(1 for x in modes if x['d_twin_mm'] < tol and x['d_mesh_mm'] >= tol)
    b = sum(1 for x in modes if x['d_twin_mm'] < tol and x['d_mesh_mm'] < tol); n = sum(1 for x in modes if x['d_twin_mm'] >= tol and x['d_mesh_mm'] >= tol); return f'{m}/{t}/{b}/{n}'
rows = []
for f in sorted(glob.glob('out/audit_ep*.json'), key=lambda s: int(s.split('ep')[-1][:-5])):
    for r in json.load(open(f)):
        print(f"{r['ep']:>2} {r['obj']:18s} sub {r['sub_md5']} v6 {r['v6_md5']} {'SAME' if r['same_file'] else 'diff'} Tcam {ang(r['Tcam']):5.1f} Ticp {ang(r['Ticp']):5.1f} sd_icp {r['symdist_icp_cm']:.3f} sd_twin {r.get('twin_icp_symdist_cm', float('nan')):.3f} selftwin {r['selftwin_symdist_cm']:.3f} v6self {r['v6_selftwin_symdist_cm']:.3f} | balls sub m/t/both/none {expl(r['balls_sub_modes'])} v6 {expl(r['balls_v6_modes'])} | frac sub {r['balls_sub']['frac_mesh_closer'] if r['balls_sub'] else None:} / {r['balls_sub']['frac_twin_closer'] if r['balls_sub'] else None} v6 {r['balls_v6']['frac_mesh_closer'] if r['balls_v6'] else None} / {r['balls_v6']['frac_twin_closer'] if r['balls_v6'] else None}")

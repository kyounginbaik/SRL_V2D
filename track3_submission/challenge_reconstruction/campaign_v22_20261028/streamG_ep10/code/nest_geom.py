"""Geometric check of the target nested pose: beige surface radius vs blue inner radius along the blue axis."""
import json, numpy as np, trimesh, pandas as pd, sys
from scipy.spatial.transform import Rotation as R
T='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5/episode_000010'
rep=json.load(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10/world_v1/recon_cup_thin_wall.json'))['report']
tb=np.load(f'{T}/blue_cup_trajectory.npz')['T_world_obj']; te=np.load(f'{T}/beige_cup_trajectory.npz')['T_world_obj']
vb=np.asarray(trimesh.load(f'{T}/blue_cup_mesh_metric.obj',force='mesh').vertices); ve=np.asarray(trimesh.load(f'{T}/beige_cup_mesh_metric.obj',force='mesh').vertices)
ab=np.array(rep['blue_cup']['axis_body']); ab/=np.linalg.norm(ab)
# blue profile: heights along ab measured from blue bottom
hb=vb@ab; hb0=hb.min(); cb=vb.mean(0)  # axis passes through? use radial fit
def radial(v, a, c): d=v-c; return np.linalg.norm(d-np.outer(d@a,a),axis=1)
# find axis center: least squares on xy of circle -> approximate by centroid of mesh projected
print('blue height', hb.max()-hb0)
H=np.array(rep['blue_cup']['band_heights_cm'])/100; IR=np.array(rep['blue_cup']['inner_radius_cm'])/100
def at(Tb,Te):
    Rel=np.linalg.inv(Tb)@Te; p=ve@Rel[:3,:3].T+Rel[:3,3]; h=p@ab-hb0; r=radial(p,ab,cb)
    ins=h<=(hb.max()-hb0)  # beige points below blue rim
    ir=np.interp(h,H,IR); pen=(r-ir)[ins & (h>H[0])]
    return h.min(), (hb.max()-hb0)-h.min(), np.percentile(pen,99) if len(pen) else np.nan, (pen>0.002).mean() if len(pen) else 0
for f in [100,120,140,160]:
    hmin,depth,p99,frac=at(tb[f],te[f]); print(f'target f{f}: beige bottom height above blue bottom {hmin*100:.1f} cm, insertion {depth*100:.1f} cm, 99pct radial overshoot {p99*100:.2f} cm, frac>2mm {frac:.2f}')
if len(sys.argv)>1:
    r=pd.read_parquet(sys.argv[1]); r=r[r.environment_id==r.environment_id.min()]
    def tr(n):
        x=r[r.object_body_name==n].sort_values('step'); Tm=np.tile(np.eye(4),(len(x),1,1))
        Tm[:,:3,3]=x[['achieved_position_x','achieved_position_y','achieved_position_z']].values
        Tm[:,:3,:3]=R.from_quat(x[['achieved_quaternion_x','achieved_quaternion_y','achieved_quaternion_z','achieved_quaternion_w']].values).as_matrix(); return Tm
    Ab,Ae=tr('blue_cup'),tr('beige_cup')
    for f in [100,160]:
        hmin,depth,p99,frac=at(Ab[f],Ae[f]); print(f'achieved f{f}: bottom {hmin*100:.1f} insertion {depth*100:.1f} overshoot {p99*100:.2f} frac {frac:.2f}')

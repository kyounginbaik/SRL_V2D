"""relative beige-in-blue pose: target_final_v5 vs a build-v2 results parquet (achieved). usage: rel_diag.py RESULTS_PARQUET"""
import sys, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
T='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5/episode_000010'
tb=np.load(f'{T}/blue_cup_trajectory.npz')['T_world_obj']; te=np.load(f'{T}/beige_cup_trajectory.npz')['T_world_obj']
r=pd.read_parquet(sys.argv[1]); r=r[r.environment_id==r.environment_id.min()]
def tr(n):
    x=r[r.object_body_name==n].sort_values('step'); T=np.tile(np.eye(4),(len(x),1,1))
    T[:,:3,3]=x[['achieved_position_x','achieved_position_y','achieved_position_z']].values
    T[:,:3,:3]=R.from_quat(x[['achieved_quaternion_x','achieved_quaternion_y','achieved_quaternion_z','achieved_quaternion_w']].values).as_matrix(); return T
ab, ae = tr('blue_cup'), tr('beige_cup'); n=min(len(ab),len(tb)); print('frames ach',len(ab),'target',len(tb))
rel=lambda A,B: np.linalg.inv(A)@B
for f in range(0,n,10):
    rt=rel(tb[f],te[f]); ra=rel(ab[f],ae[f])
    # mesh z axis = cup axis? print rel translation in blue frame + world z of each
    print(f'f{f:3d} tgt rel {np.round(rt[:3,3]*100,1)} ach rel {np.round(ra[:3,3]*100,1)} | tgt z b/e {tb[f,2,3]*100:6.1f} {te[f,2,3]*100:6.1f}  ach z b/e {ab[f,2,3]*100:6.1f} {ae[f,2,3]*100:6.1f}  tilt-rel tgt {np.degrees(np.arccos(np.clip(rt[2,2],-1,1))):5.1f} ach {np.degrees(np.arccos(np.clip(ra[2,2],-1,1))):5.1f}')

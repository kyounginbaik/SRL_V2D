import sys, json, numpy as np, pandas as pd, os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
src=Path(sys.argv[1]); out=Path(sys.argv[2]); (out/'results').mkdir(parents=True,exist_ok=True)
G=np.array(json.load(open('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json'))['G'])
r=pd.read_parquet(src/'results/episode_000019.parquet'); m=r.object_body_name=='mini_sweeper'
q=r.loc[m,[f'achieved_quaternion_{a}' for a in 'xyzw']].to_numpy(); p=r.loc[m,[f'achieved_position_{a}' for a in 'xyz']].to_numpy()
T=np.tile(np.eye(4),(len(q),1,1)); T[:,:3,:3]=R.from_quat(q).as_matrix(); T[:,:3,3]=p; T2=T@G
r.loc[m,[f'achieved_position_{a}' for a in 'xyz']]=T2[:,:3,3]; r.loc[m,[f'achieved_quaternion_{a}' for a in 'xyzw']]=R.from_matrix(T2[:,:3,:3]).as_quat()
r.to_parquet(out/'results/episode_000019.parquet',index=False)
os.symlink(src/'meshes', out/'meshes')

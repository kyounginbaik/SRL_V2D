"""Print sweeper tilt (long axis vs vertical) and slot xy offset at the end for one or more runs (dirs with result.json)."""
import sys, json, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
V='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9'
z=np.load(V+'/collision_recon/episode_000030/object_poses.npz'); T=z['world_from_object']; i=[str(x) for x in z['names']].index('mini_sweeper')
v=np.asarray(trimesh.load(V+'/collision_recon/episode_000030/assets/mini_sweeper_visual.obj',force='mesh').vertices); c=v-v.mean(0); w,U=np.linalg.eigh(c.T@c); L=U[:,-1]; ctr=v.mean(0)
ts=(11.0,11.8,12.2,12.6,13.1)
def row(Rs,P,tag):
    k=lambda t:min(int(t*20),len(P)-1); kr=lambda t:min(int(round(11.9*20)),len(T)-1)   # slot location = reference just before the 12.2 s track jump
    tilt=[np.degrees(np.arccos(abs((Rs[k(t)]@L)[2]))) for t in ts]; xy=np.linalg.norm((Rs[-1]@ctr+P[-1])[:2]-(T[kr(0),i,:3,:3]@ctr+T[kr(0),i,:3,3])[:2])*100
    print(f'{tag}: tilt ' + ' '.join(f'{t}:{a:.0f}' for t,a in zip(ts,tilt)) + f' | end xy from slot {xy:.1f} cm'); return tilt[-1],xy
for d in sys.argv[1:]:
    r=json.load(open(d+'/result.json')); a=np.load(r['execution']['trajectory']); j=list(a['object_names']).index('mini_sweeper')
    m=r['metrics']['official_style_reference_proxy']['objects']['mini_sweeper']; row(R.from_quat(a['object_quaternions_wxyz'][:,j][:,[1,2,3,0]]).as_matrix(),a['object_positions'][:,j],f"{d.split('/')[-1]} mp_sr {r['metrics']['reference_proxy_mp_sr']:.0f} sw {m['mean_position_error_cm'][0]:.2f}/{m['mean_orientation_error_degrees'][0]:.1f}")

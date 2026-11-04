import numpy as np, json, sys, trimesh
from scipy.spatial.transform import Rotation as R
W='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021'
v=np.asarray(trimesh.load(W+'/collision_recon/episode_000014/assets/blue_cup_visual.obj',force='mesh').vertices); c=v-v.mean(0); w,U=np.linalg.eigh(c.T@c)
g=[abs(w[k]-w[(k+1)%3])+abs(w[k]-w[(k+2)%3])-abs(w[(k+1)%3]-w[(k+2)%3]) for k in range(3)]; ax=U[:,int(np.argmax(g))]
tilt=lambda M: np.degrees(np.arccos(abs((M@ax)[2])))
z=np.load(W+'/collision_recon/episode_000014/object_poses.npz'); T=z['world_from_object']; i=[str(x) for x in z['names']].index('blue_cup')
for p in sys.argv[1:]:
    r=json.load(open(p+'/result.json')); a=np.load(r['execution']['trajectory']); j=list(a['object_names']).index('blue_cup'); Rs=R.from_quat(a['object_quaternions_wxyz'][:,j][:,[1,2,3,0]]).as_matrix(); P=a['object_positions'][:,j]
    o=r['metrics']['official_style_reference_proxy']['objects']['blue_cup']
    print(p.split('rackscan_20261021/')[-1], 'mp_sr', r['metrics']['reference_proxy_mp_sr'], 'cup %.2f/%.1f'%(o['mean_position_error_cm'][0],o['mean_orientation_error_degrees'][0]), '| tilt', ' '.join(f'{t}:{tilt(Rs[min(int(t*20),len(Rs)-1)]):.0f}' for t in (4,5,6,6.5,7,7.8)), '| final z %.1f (ref %.1f)'%(P[-1,2]*100,T[-1,i,2,3]*100))

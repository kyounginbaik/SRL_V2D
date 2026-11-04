"""per-second object errors (body origin, setup frame) of one or more actor dirs vs the original reference. usage: windows.py DIR [DIR..]"""
import json, sys, numpy as np
from scipy.spatial.transform import Rotation as R
EPD='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033'
S=np.array(json.load(open(EPD+'/input_manifest.json'))['setup_from_source'])
z=np.load('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000033/object_poses.npz'); P=S@z['world_from_object']
res=[]
for d in sys.argv[1:]:
    tr=np.load(json.load(open(d+'/evaluation/result.json'))['execution']['trajectory']); row=[]
    for o in range(2):
        e=np.linalg.norm(tr['object_positions'][:,o]-P[:,o,:3,3],axis=1)*100
        a=np.degrees((R.from_quat(tr['object_quaternions_wxyz'][:,o][:,[1,2,3,0]])*R.from_matrix(P[:,o,:3,:3]).inv()).magnitude())
        row.append((e,a))
    res.append((d.rstrip('/').split('/')[-1],row))
print('  t   ' + ' | '.join(f'{n[:22]:>22s}' for n,_ in res))
for s in range(0,348,20):
    print(f'{s/20:4.0f}  ' + ' | '.join(f'c {r[0][0][s:s+20].mean():4.1f}/{r[0][1][s:s+20].mean():4.0f} s {r[1][0][s:s+20].mean():4.1f}/{r[1][1][s:s+20].mean():4.0f}' for _,r in res))

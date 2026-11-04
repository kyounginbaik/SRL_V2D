"""Board 2 centre in board 1's PCA frame (thin,mid,long cm), angle between long axes, heights: REF vs each result dir, 12.5-17.75 s.
usage: insertion_geom.py RESULT_DIR [RESULT_DIR ...]"""
import sys,numpy as np,mujoco,json,trimesh
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from scipy.spatial.transform import Rotation as R
EP='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025'
A='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025/assets'
U={}
for o in ('wooden_piece_1','wooden_piece_2'):
    v=np.asarray(trimesh.load(A+f'/{o}_visual.obj',force='mesh').vertices); c0=v.mean(0); w,UU=np.linalg.eigh((v-c0).T@(v-c0)); U[o]=UU
runs=[]
for d in sys.argv[1:]:
    r=json.load(open(d+'/result.json')); runs.append((d.rstrip('/').split('/')[-1],np.load(r['execution']['trajectory']),r['execution']['scene']))
m=mujoco.MjModel.from_xml_path(runs[0][2]); ref=TR.load_reference(EP+'/tracking_reference.npz',m)['qpos']
def pose(tr,k,o):
    j=list(tr['object_names']).index(o); return tr['object_positions'][k,j], R.from_quat(tr['object_quaternions_wxyz'][k,j][[1,2,3,0]]).as_matrix()
def refpose(k,o):
    oc=int(m.joint(o+'_free').qposadr[0]); return ref[k,oc:oc+3], R.from_quat(ref[k,oc+3:oc+7],scalar_first=True).as_matrix()
def row(p1,R1,p2,R2):
    F1=R1@U['wooden_piece_1']; rel=F1.T@(p2-p1); ang=np.degrees(np.arccos(abs(F1[:,2]@(R2@U['wooden_piece_2'])[:,2])))
    return f'[{rel[0]*100:5.1f} {rel[1]*100:5.1f} {rel[2]*100:5.1f}] {ang:3.0f}deg z {p1[2]*100:4.1f}/{p2[2]*100:4.1f}'
print('t      REF'+''.join(f' | {n[:22]:22s}' for n,_,_ in runs))
for k in range(250,356,10):
    s=f'{k/20:5.1f} '+row(*refpose(k,'wooden_piece_1'),*refpose(k,'wooden_piece_2'))
    for n,tr,_ in runs: s+=' | '+row(*pose(tr,k,'wooden_piece_1'),*pose(tr,k,'wooden_piece_2'))
    print(s)

"""Official metric from a canonical run: per-object episode means with first-pose alignment."""
import sys,json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
run=Path(sys.argv[1]);z=np.load(run/'physics.npz');res=json.loads((run/'result.json').read_text());g=z['world_reference_group'];P=z['object_pose']
print('steps',res['steps'],'of',res['reference_frames']-1,'full',res['full_episode'],'official_mp_sr',res['official_mp_sr'],'assistance',res['assistance'],'hash',res['task_commands_sha256'][:10])
for c in res['candidates']:print(' candidate',c)
for w in range(P.shape[1]):
    for ob in range(P.shape[2]):
        o=P[:,w,ob];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();off=i[:3]-al.apply(i[7:10]);gp=al.apply(o[:,7:10])+off;gr=al*R.from_quat(o[:,10:14]);ep=np.linalg.norm(o[:,:3]-gp,axis=1);er=np.degrees((R.from_quat(o[:,3:7]).inv()*gr).magnitude())
        print(' world',w,'group',int(g[w]),'obj',ob,'mean',round(100*ep.mean(),2),'cm',round(er.mean(),1),'deg','| max',round(100*ep.max(),1),'cm',round(er.max(),0),'deg','| end',round(100*ep[-1],1),'cm')

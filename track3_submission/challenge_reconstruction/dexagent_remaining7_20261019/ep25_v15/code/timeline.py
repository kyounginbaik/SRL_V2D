import numpy as np, json, sys
from scipy.spatial.transform import Rotation as R
for run in sys.argv[1:]:
    r=json.load(open(run+'/result.json')); a=np.load(r['execution']['trajectory']); src=r['spec']['source'] if r.get('spec') else json.load(open(r['input_dir']+'/input_manifest.json'))['source']
    T=np.load(src+'/object_poses.npz')['world_from_object']; print(run.split('/')[-1])
    for j,n in enumerate(a['object_names']):
        s=[]
        for t in range(0,18):
            k=min(t*20,len(a['object_positions'])-1); p=a['object_positions'][k,j]; q=R.from_quat(a['object_quaternions_wxyz'][k,j][[1,2,3,0]])
            s.append(f"{t}:{np.linalg.norm(p-T[k,j,:3,3])*100:.0f}/{np.degrees((q*R.from_matrix(T[k,j,:3,:3]).inv()).magnitude()):.0f}/z{p[2]*100:.0f}")
        print(' ',n,' '.join(s))

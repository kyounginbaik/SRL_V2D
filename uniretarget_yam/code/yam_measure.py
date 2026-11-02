import json,re,numpy as np,mujoco
from pathlib import Path
E=Path(__file__).parent.parent/'episodes/episode_000037/yam'
m=mujoco.MjModel.from_binary_path(str(E/'yam_model.mjb'));p=json.loads((E/'yam_placement.json').read_text());up=np.array(p['up']);d=mujoco.MjData(m)
READY=[0.,1.536,.945,-.61,0.,0.]
for j in range(m.njnt):
    k=re.search(r'_arm_joint([1-6])$',m.joint(j).name)
    if k:d.qpos[m.jnt_qposadr[j]]=READY[int(k.group(1))-1]
mujoco.mj_forward(m,d);q0=d.qpos.copy()
pm={s:d.xpos[m.body(f'{s}_arm_{s}_hand_C_MC').id].copy() for s in('left','right')}
for s,o in(('left','right'),('right','left')):
    b=m.body(f'{s}_arm_{s}_hand_C_MC').id;R=d.xmat[b].reshape(3,3).copy();inward=pm[o]-pm[s];inward/=np.linalg.norm(inward);flex=[]
    print(s,'palm world',pm[s].round(3).tolist(),'base',d.xpos[m.body(f'{s}_arm_base_link').id].round(3).tolist())
    for f in('index','middle','ring','pinky'):
        js=[j for j in range(m.njnt) if m.joint(j).name.startswith(f'{s}_arm_{s}_{f}') and re.search('MCP_FE|PIP|DIP',m.joint(j).name)]
        t=m.body(f'{s}_arm_{s}_{f}_fingertip').id;a=d.xpos[t].copy()
        for j in js:d.qpos[m.jnt_qposadr[j]]=min(.6,m.jnt_range[j][1])
        mujoco.mj_forward(m,d);v=d.xpos[t]-a;v/=np.linalg.norm(v);d.qpos[:]=q0;mujoco.mj_forward(m,d);flex.append(v)
        print('  ',f,'ranges',[m.jnt_range[j].round(2).tolist() for j in js][:1],'flex world',v.round(2).tolist())
    n=np.mean(flex,0);n/=np.linalg.norm(n);th=d.xpos[m.body(f'{s}_arm_{s}_thumb_fingertip').id]-pm[s];ix=d.xpos[m.body(f'{s}_arm_{s}_index_fingertip').id]-pm[s]
    print('  PALM NORMAL world',n.round(2).tolist(),'in palm frame',(R.T@n).round(2).tolist(),'dot inward',round(float(n@inward),2),'dot up',round(float(n@up),2))
    print('  thumb tip from palm, palm frame',(R.T@th).round(3).tolist(),'index tip',(R.T@ix).round(3).tolist(),'handed (fingers x thumb)·n',round(float(np.cross(ix,th)@n),4))

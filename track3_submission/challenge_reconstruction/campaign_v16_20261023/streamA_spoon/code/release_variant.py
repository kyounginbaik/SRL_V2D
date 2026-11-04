"""Release re-plan on a teacher reference: from t_rel the carrying hand's finger flexion joints open to open_frac of range over
open_dur seconds while the wrist is frozen at its t_rel pose until t_lift; after t_lift the wrist blends back onto the reference
path over blend_dur. Objects, clock and the other hand untouched. Chain on top of any reference dir (result.json + reference.npz).
usage: release_variant.py --teacher DIR --episode-dir DIR --out DIR [--side right --object wooden_spoon --t-rel 19.0 --open-dur 0.3
       --open-frac 0.1 --t-lift 19.6 --blend-dur 0.6 --fingers thumb,index,middle,ring,pinky]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
ROOT=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(ROOT/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
sys.path.insert(0,str(Path(__file__).resolve().parent)); from curl_variant import inhand, dump
sm=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
FLEX={'thumb':('CMC_FE','MCP_FE','IP'),'index':('MCP_FE','PIP','DIP'),'middle':('MCP_FE','PIP','DIP'),'ring':('MCP_FE','PIP','DIP'),'pinky':('MCP_FE','PIP','DIP')}

def build(m,Q0,fps,a):
    T=len(Q0); t=np.arange(T)/fps; Q=Q0.copy(); pre=f'robot_{a.side}_{a.side}_'
    wo=sm((t-a.t_rel)/a.open_dur)                      # finger opening weight
    for f in a.fingers.split(','):
        for jn in FLEX[f]:
            j=m.joint(pre+f+'_'+jn).id; adr=int(m.jnt_qposadr[j]); lo,hi=m.jnt_range[j]; tgt=lo+a.open_frac*(hi-lo)
            Q[:,adr]=(1-wo)*Q0[:,adr]+wo*np.minimum(Q0[:,adr],tgt)
    k_rel=int(round(a.t_rel*fps)); wc=[int(m.joint(pre+'pos_'+x).qposadr[0]) for x in 'xyz']; wr=[int(m.joint(pre+'rot_'+x).qposadr[0]) for x in 'xyz']
    hold=np.clip((t-a.t_rel)/1e-9,0,1)*(1-sm((t-a.t_lift)/a.blend_dur))   # 1 during [t_rel, t_lift], blends to 0 by t_lift+blend
    hold[t<a.t_rel]=0
    p0=Q0[k_rel,wc]; r0=R.from_euler('XYZ',Q0[k_rel,wr])
    for k in range(T):
        h=hold[k]
        if h<=0: continue
        Q[k,wc]=(1-h)*Q0[k,wc]+h*p0
        rk=R.from_euler('XYZ',Q0[k,wr]); rel=(r0*rk.inv()).as_rotvec(); Q[k,wr]=continuous_euler_xyz(R.from_matrix((R.from_rotvec(h*rel)*rk).as_matrix()[None]),Q0[k,wr])[0]
    return Q

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--teacher',required=True); p.add_argument('--episode-dir',required=True); p.add_argument('--out',required=True)
    p.add_argument('--side',default='right'); p.add_argument('--object',default='wooden_spoon'); p.add_argument('--fingers',default='thumb,index,middle,ring,pinky')
    for k,v in (('t_rel',19.),('open_dur',.3),('open_frac',.1),('t_lift',19.6),('blend_dur',.6)): p.add_argument('--'+k.replace('_','-'),type=float,default=v)
    a=p.parse_args(); src=Path(a.teacher); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); epd=Path(a.episode_dir)
    prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
    ref=TR.load_reference(src/'reference.npz',m); Q0=ref['qpos']; fps=ref['fps']; Q=build(m,Q0,fps,a)
    lim=m.jnt_range[m.actuator_trnid[:,0]]; qa=m.jnt_qposadr[m.actuator_trnid[:,0]]; Q[:,qa]=np.clip(Q[:,qa],lim[:,0],lim[:,1]); Q[0]=Q0[0]
    refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'release re-plan {vars(a)} on {src}; immutable object targets/clock',arms_coupled=False,floating_runtime=True)
    ex=execute_reference(scene,refp,out/'execution'); man=json.loads((epd/'input_manifest.json').read_text())
    met=evaluate_files(man['source'],ex['trajectory'],epd/'input_manifest.json',ex['scene'],epd/'metric_geometry.json'); ih=inhand(m,np.load(ex['trajectory']),a.object,a.side)
    dump(out/'result.json',dict(episode=prev.get('episode',man.get('episode')),input_dir=str(epd),reference=str(refp),execution=ex,metrics=met,inhand=ih,variant=vars(a),source=str(src),learned_policy=False))
    o=met['official_style_reference_proxy']['objects']
    print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()},inhand=ih,completed=ex['execution_completed'])),flush=True)

"""Blend a constant world-frame wrist translation offset (and optional rotation about the wrist) onto one hand during a
window; used e.g. to move the idle hand clear of the carried object. Chainable (reads result.json + reference.npz).
usage: wrist_offset.py --teacher DIR --episode-dir DIR --out DIR --side left --t-on 3.0 --t-off 9.5 [--ramp 0.5 --dx 0 --dy 0 --dz 0.08 --object wooden_spoon --score-side right]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
ROOT=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(ROOT/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
sys.path.insert(0,str(Path(__file__).resolve().parent)); from curl_variant import inhand, dump
sm=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--teacher',required=True); p.add_argument('--episode-dir',required=True); p.add_argument('--out',required=True)
    p.add_argument('--side',default='left'); p.add_argument('--object',default='wooden_spoon'); p.add_argument('--score-side',default='right')
    for k,v in (('t_on',3.),('t_off',9.5),('ramp',.5),('dx',0.),('dy',0.),('dz',.08)): p.add_argument('--'+k.replace('_','-'),type=float,default=v)
    a=p.parse_args(); src=Path(a.teacher); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); epd=Path(a.episode_dir)
    prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
    ref=TR.load_reference(src/'reference.npz',m); Q0=ref['qpos']; fps=ref['fps']; T=len(Q0); t=np.arange(T)/fps
    w=sm((t-a.t_on)/a.ramp)*(1-sm((t-a.t_off)/a.ramp)); Q=Q0.copy(); wc=[int(m.joint(f'robot_{a.side}_{a.side}_pos_'+x).qposadr[0]) for x in 'xyz']
    Q[:,wc]=Q0[:,wc]+w[:,None]*np.array([a.dx,a.dy,a.dz]); Q[0]=Q0[0]
    refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'wrist offset {vars(a)} on {src}; immutable object targets/clock',arms_coupled=False,floating_runtime=True)
    ex=execute_reference(scene,refp,out/'execution'); man=json.loads((epd/'input_manifest.json').read_text())
    met=evaluate_files(man['source'],ex['trajectory'],epd/'input_manifest.json',ex['scene'],epd/'metric_geometry.json'); ih=inhand(m,np.load(ex['trajectory']),a.object,a.score_side)
    dump(out/'result.json',dict(episode=prev.get('episode',man.get('episode')),input_dir=str(epd),reference=str(refp),execution=ex,metrics=met,inhand=ih,variant=vars(a),source=str(src),learned_policy=False))
    o=met['official_style_reference_proxy']['objects']
    print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()},inhand=ih,completed=ex['execution_completed'])),flush=True)

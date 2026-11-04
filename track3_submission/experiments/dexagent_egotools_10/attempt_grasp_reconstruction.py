"""Bounded, native Stage 2 attempt after the verified grasp Stage 1 delivery."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parent
DEX=ROOT.parents[1]/'third_party/DexAgent'
sys.path.insert(0,str(DEX))
from agent.external import atomic_json

ident='01_grasp_test_tube'
cfg=yaml.safe_load((ROOT/'configs'/(ident+'.yaml')).read_text())
transport=ROOT/'reconstruction_transport'/ident
cfg['agent']['transport_dir']=str(transport)
cfg['agent']['response_timeout_s']=90
cfg['stage2']['max_rounds']=4
config=ROOT/'configs'/'01_grasp_reconstruction.yaml'
config.write_text(yaml.safe_dump(cfg))
video=ROOT/'clips'/(ident+'.mp4')
code=f'''from pathlib import Path
import json
from tool_library.skills.extract_frames import extract_frames
from agent.config import load_config
from agent.library import call_tool, load_index
meta=extract_frames({str(video)!r}, 'frames', stride=10, start=375, max_frames=8)
print(json.dumps(meta))
cfg=load_config({str(config)!r})
lib=Path(cfg['inputs']['library'])
result=call_tool(lib, load_index(lib), 'detect_objects', {{'frames_dir':str(Path('frames').resolve()), 'keywords':['transparent test tube','red rack'], 'out_json':str(Path('detections.json').resolve())}}, cfg, cwd=Path.cwd())
Path('perception_attempt.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''
log=(ROOT/'logs'/'01_grasp_reconstruction.log').open('w')
child=subprocess.Popen([sys.executable,str(DEX/'stage2_scene_reconstruction/run_stage2.py'),str(config)],
    cwd=DEX,stdout=log,stderr=subprocess.STDOUT,
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl'))
deadline=time.monotonic()+150
seen=set()
while child.poll() is None and time.monotonic()<deadline:
    p=transport/'status.json'
    s=json.loads(p.read_text()) if p.exists() else {}
    if s.get('state')=='waiting' and s['id'] not in seen:
        if s['turn']==1:
            content=[{'type':'tool_use','id':'try_perception','name':'run_python','input':{'code':code,'timeout_s':60}}]
        elif s['turn']==2:
            content=[{'type':'tool_use','id':'check_reconstruction','name':'finish','input':{'summary':'Expose the actual reconstruction verification verdict after the perception dependency attempt.'}}]
        else:
            content=[{'type':'tool_use','id':'report_environment','name':'report_blocked','input':{'reason':'Required source object detection failed in the configured interpreter; see perception_attempt.json. No verified scene or source masks were produced. The selected video also has a moving head camera, while the supplied camera contract is static. This attempt does not establish robot task success.'}}]
        atomic_json(transport/'responses'/(s['id']+'.json'),{'id':s['id'],'session':s['session'],'content':content})
        seen.add(s['id'])
    time.sleep(.15)
if child.poll() is None:child.terminate()
rc=child.wait(timeout=10)
log.close()
print(json.dumps({'returncode':rc,'log':str(ROOT/'logs'/'01_grasp_reconstruction.log')}))
# A native failure is an observed test result, rather than a successful reconstruction.
raise SystemExit(rc)

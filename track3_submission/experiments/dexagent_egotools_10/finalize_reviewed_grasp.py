"""Deliver the reviewed grasp candidate through the native external transport.

All model decisions and semantic text are supplied by the current Codex session.
This bounded replay closes an interrupted native session; it is not a new model
evaluation or an automatic semantic-answer generator.
"""
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

ident=sys.argv[1] if len(sys.argv)>1 else '01_grasp_test_tube'
cfg=yaml.safe_load((ROOT/'configs'/(ident+'.yaml')).read_text())
transport=ROOT/'delivery_transport'/ident
cfg['agent']['transport_dir']=str(transport)
cfg['agent']['response_timeout_s']=90
cfg['stage1']['max_rounds']=3
config=ROOT/'configs'/(ident+'_delivery.yaml')
config.write_text(yaml.safe_dump(cfg))
log_path=ROOT/'logs'/(ident+'_delivery.log')
log=log_path.open('w')
child=subprocess.Popen([sys.executable,str(DEX/'stage1_semantic_understanding/run_stage1.py'),str(config)],
    cwd=DEX,stdout=log,stderr=subprocess.STDOUT,
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl'))
deadline=time.monotonic()+120
seen=set()
while child.poll() is None and time.monotonic()<deadline:
    p=transport/'status.json'
    s=json.loads(p.read_text()) if p.exists() else {}
    if s.get('state')=='waiting' and s['id'] not in seen:
        req=json.loads(Path(s['request']).read_text())
        if s['turn']==1:
            content=[{'type':'tool_use','id':'reviewed_document','name':'write_text',
                'input':{'path':'scene_semantics.json','text':(ROOT/'reviewed_semantics'/(ident+'.json')).read_text()}},
                {'type':'tool_use','id':'native_finish','name':'finish','input':{'summary':'Deliver the current-session video-reviewed candidate with all native checks.'}}]
        else:
            content=[{'type':'tool_use','id':'native_block','name':'report_blocked','input':{'reason':'The reviewed candidate failed the native finish checks; preserve their exact verdict.'}}]
        atomic_json(transport/'responses'/(s['id']+'.json'),{'id':s['id'],'session':s['session'],'content':content})
        seen.add(s['id'])
    time.sleep(.15)
if child.poll() is None:
    child.terminate()
rc=child.wait(timeout=10)
log.close()
print(json.dumps({'returncode':rc,'log':str(log_path)}))
raise SystemExit(rc)

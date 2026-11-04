"""Operator utility for the unmodified DexAgent external-model transport.

The current Codex session supplies tool calls; this utility makes no model calls
and supplies no scripted semantic answers. Requests and responses remain on disk.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEX = ROOT.parents[1] / 'third_party/DexAgent'
sys.path.insert(0, str(DEX))
from agent.external import atomic_json, validate_response

def status(task):
    p = ROOT / 'transport' / task / 'status.json'
    return json.loads(p.read_text()) if p.exists() else {}

parser = argparse.ArgumentParser()
parser.add_argument('command',choices=['start','status','inspect','respond'])
parser.add_argument('--task')
parser.add_argument('--stage', default='stage1',choices=['stage1','stage2'])
parser.add_argument('--file')
args = parser.parse_args()
tasks = json.loads((ROOT/'tasks.json').read_text())
selected = [t for t in tasks if not args.task or t['id']==args.task]
if args.command=='start':
    children = []
    for t in selected:
        if not (ROOT/'configs'/(t['id']+'.yaml')).exists() or (ROOT/'transport'/t['id']/'status.json').exists():
            if args.stage=='stage1':
                continue
        folder = ROOT/'logs'
        folder.mkdir(exist_ok=True)
        log = (folder/(t['id']+'_'+args.stage+'.log')).open('w')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl')
        script = DEX / ('stage1_semantic_understanding/run_stage1.py' if args.stage=='stage1' else 'stage2_scene_reconstruction/run_stage2.py')
        p = subprocess.Popen([sys.executable,str(script),str(ROOT/'configs'/(t['id']+'.yaml'))],
                              cwd=DEX, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        children.append(p)
        print(json.dumps({'task':t['id'],'pid':p.pid,'stage':args.stage}),flush=True)
    for child in children:
        child.wait()
elif args.command=='status':
    for t in selected:
        s=status(t['id'])
        print(json.dumps({'task':t['id'], **s}),flush=True)
elif args.command=='inspect':
    for t in selected:
        s=status(t['id'])
        if s.get('state')!='waiting':
            print(json.dumps({'task':t['id'],**s}));continue
        request=json.loads(Path(s['request']).read_text())
        # Read the latest feedback, without repeating prior messages or images.
        print(json.dumps({'task':t['id'],'turn':s['turn'],'system':request['system'] if args.file=='system' else None,
                          'messages':request['messages'][-1:],
                          'tools':[x['name'] for x in request['tools']]}))
else:
    responses=json.loads(Path(args.file).read_text())
    for t in selected:
        if t['id'] not in responses:continue
        s=status(t['id'])
        assert s['state']=='waiting', (t['id'],s)
        req=json.loads(Path(s['request']).read_text())
        blocks=validate_response(responses[t['id']],{x['name'] for x in req['tools']})
        dest=ROOT/'transport'/t['id']/'responses'/(s['id']+'.json')
        assert not dest.exists(), str(dest)
        atomic_json(dest,{'id':s['id'],'session':s['session'],'content':blocks})
        print(json.dumps({'submitted':t['id'],'turn':s['turn']}),flush=True)

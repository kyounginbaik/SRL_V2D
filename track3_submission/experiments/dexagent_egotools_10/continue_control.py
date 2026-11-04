"""Inspect and answer the native external transport for continued reconstruction."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent
DEX=ROOT/'deps/DexAgent-adapted'
sys.path.insert(0,str(DEX))
from agent.external import atomic_json,validate_response

p=argparse.ArgumentParser()
p.add_argument('command',choices=['inspect','respond'])
p.add_argument('--task',default='grasp')
p.add_argument('--stage',default='stage2')
p.add_argument('--file')
a=p.parse_args()
transport=ROOT/'continuation/transport'/a.task/a.stage
s=json.loads((transport/'status.json').read_text())
if a.command=='inspect':
    print(json.dumps(s))
    if s['state']=='waiting':
        q=json.loads(Path(s['request']).read_text())
        print(json.dumps({'messages':q['messages'][-1:],'tools':[x['name'] for x in q['tools']]}))
else:
    assert s['state']=='waiting',s
    q=json.loads(Path(s['request']).read_text())
    blocks=json.loads(Path(a.file).read_text())
    validate_response(blocks,{x['name'] for x in q['tools']})
    dest=transport/'responses'/(s['id']+'.json')
    assert not dest.exists(),str(dest)
    atomic_json(dest,{'session':s['session'],'id':s['id'],'content':blocks})
    print(json.dumps({'submitted':s['id']}))

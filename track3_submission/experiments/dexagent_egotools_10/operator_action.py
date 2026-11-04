"""Submit an explicit operator-selected tool action to waiting external requests.

Semantic documents are authored by the current Codex session after image review.
This does not implement a replacement model or generate semantic documents.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'third_party/DexAgent'))
from agent.external import atomic_json, validate_response

p=argparse.ArgumentParser()
p.add_argument('action', choices=['overview','look','submit','finish','blocked'])
p.add_argument('--task')
p.add_argument('--reason')
a=p.parse_args()
for task in json.loads((ROOT/'tasks.json').read_text()):
    if a.task and a.task!=task['id']:continue
    transport=ROOT/'transport'/task['id']
    if not (transport/'status.json').exists():continue
    s=json.loads((transport/'status.json').read_text())
    if s.get('state')!='waiting':continue
    req=json.loads(Path(s['request']).read_text())
    dest=transport/'responses'/(req['id']+'.json')
    if dest.exists():continue
    blocks=[]
    def tool(name,inp):
        blocks.append(dict(type='tool_use',id=f'{a.action}_{len(blocks)}',name=name,input=inp))
    if a.action=='overview':
        if s['turn']!=1:continue
        clip=str(ROOT/'clips'/(task['id']+'.mp4'))
        tool('run_python',{'code':f'from tool_library.skills.probe_video import probe_video\nfrom tool_library.skills.make_contact_sheet import make_contact_sheet\nimport json\nprint(json.dumps(probe_video({clip!r})))\nprint(json.dumps(make_contact_sheet({clip!r}, "overview.jpg", count=24, columns=6, tile_width=300)))\nprint(json.dumps(make_contact_sheet({clip!r}, "detail.jpg", count=12, columns=4, tile_width=400, crop=[0.15,0.15,0.95,0.8])))','timeout_s':120})
    elif a.action=='look':
        if s['turn']!=2:continue
        tool('look',{'paths':['overview.jpg','detail.jpg']})
    elif a.action=='submit':
        if s['turn']<3:continue
        semantic=ROOT/'reviewed_semantics'/(task['id']+'.json')
        if not semantic.exists():continue
        tool('write_text',{'path':'scene_semantics.json','text':semantic.read_text()})
        tool('finish',{'summary':'Video-reviewed semantic decomposition; preserve all tool-mediated and material outcomes for native verification.'})
    elif a.action=='finish':
        tool('finish',{'summary':'Run the native stage verification on the current artifacts and expose exact compatibility failures.'})
    elif a.action=='blocked':
        assert a.reason
        tool('report_blocked',{'reason':a.reason})
    validate_response(blocks,{t['name'] for t in req['tools']})
    atomic_json(dest,{'session':req['session'],'id':req['id'],'content':blocks})
    print(json.dumps({'task':task['id'],'turn':s['turn'],'action':a.action}),flush=True)

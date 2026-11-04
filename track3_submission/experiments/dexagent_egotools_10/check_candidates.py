"""Reproduce fixed-library checks of the video-reviewed task definitions.

This measures input/schema/goal compatibility, not VLM accuracy or robot success.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
DEX=Path(os.environ.get('DEXAGENT_REPO',str(ROOT.parents[1]/'third_party/DexAgent'))).resolve()
sys.path.insert(0,str(DEX))
from agent.config import load_config
from agent.library import call_tool, load_index
from tool_library.skills.probe_video import probe_video

tasks=json.loads((ROOT/'tasks.json').read_text())
media={r['id']:r for r in json.loads((ROOT/'media_validation.json').read_text())}
lib=DEX/'tool_library'
index=load_index(lib)
out=[]
for task in tasks:
    cfg=load_config(ROOT/'configs'/(task['id']+'.yaml'))
    video=ROOT/'clips'/(task['id']+'.mp4')
    facts=probe_video(str(video))
    expected=media[task['id']]['clip_sha256']
    actual=hashlib.sha256(video.read_bytes()).hexdigest()
    candidate=ROOT/'reviewed_semantics'/(task['id']+'.json')
    checks={'clip_integrity':{'ok':actual==expected,'sha256':actual,'video_info':facts}}
    for name,args in [
        ('scene_semantic_schema_check',{'path':str(candidate)}),
        ('scene_semantic_consistency_check',{'path':str(candidate),'prompt':task['task'],'video_duration_s':facts['duration_s']}),
        ('goal_language_coverage_check',{'path':str(candidate)})]:
        result=call_tool(lib,index,name,args,cfg)
        checks[name]=result['result'] if result.get('ok') else {'ok':False,'execution_error':result.get('error')}
    row=dict(task,checks=checks,compatible=all(c.get('ok') is True for c in checks.values()))
    out.append(row)
    (ROOT/'compatibility.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'task':task['id'],'checks':{k:v.get('ok') for k,v in checks.items()}}),flush=True)

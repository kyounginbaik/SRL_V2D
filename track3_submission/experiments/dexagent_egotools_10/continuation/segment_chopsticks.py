"""Perception diagnostic only; the native task goal remains unsupported."""
from pathlib import Path
import json
import os
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps/DexAgent-adapted'))
os.environ['HF_HOME']=str(ROOT/'models/hf')
from tool_library.skills.segment_with_text_prompts import segment_with_text_prompts
out=ROOT/'continuation/chopsticks'
(out/'frames').mkdir(exist_ok=True)
shutil.copy2(out/'source_frame0.png',out/'frames/frame00000.png')
result=segment_with_text_prompts(str(out/'frames'),['grape','chopstick','bread','plate'],str(out/'sam3'),instances=True)
(out/'perception_result.json').write_text(json.dumps({'scope':'Source-derived perception diagnostic, not native Stage 2 acceptance or a robot execution.',
    'result':result},indent=2))
print(json.dumps({'n_instances':result.get('n_instances'),'model_identity':result.get('model_identity')}),flush=True)

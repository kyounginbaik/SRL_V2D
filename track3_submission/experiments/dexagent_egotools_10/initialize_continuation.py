"""Preserve the baseline and start isolated continuation artifacts."""
from pathlib import Path
import json
import shutil
import yaml

ROOT=Path(__file__).resolve().parent
DEX=ROOT/'deps/DexAgent-adapted'
CONT=ROOT/'continuation'
CONT.mkdir(exist_ok=True)
stage1=CONT/'runs/grasp/stage1'
stage1.mkdir(parents=True,exist_ok=True)
for name in ['scene_semantics.json','input_video.json','run.json']:
    target=stage1/name
    if not target.exists():shutil.copyfile(ROOT/'runs/01_grasp_test_tube/stage1'/name,target)
cfg=yaml.safe_load((ROOT/'configs/01_grasp_test_tube.yaml').read_text())
cfg['run']={'name':'grasp','output_dir':str(CONT/'runs')}
cfg['agent'].update(transport_dir=str(CONT/'transport/grasp/stage2'),tool_discovery='all',response_timeout_s=7200)
cfg['inputs'].update(setup=str(DEX/'setup'),library=str(DEX/'tool_library'))
cfg['environments']={'sam3':str(ROOT/'.perception/bin/python')}
cfg['models']={'hf_home':str(ROOT/'models/hf')}
cfg['stage2']={'max_rounds':60,'max_wall_minutes':120}
cfg['stage3']={'max_rounds':60,'max_wall_minutes':120,'methods':[],'stage4_draws':3,'strict_quality':True}
(CONT/'grasp.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(CONT/'provenance.json').write_text(json.dumps({'baseline_results':'../results.json','baseline_unchanged':True,
    'stage1_origin':'../runs/01_grasp_test_tube/stage1','same_input_video':True,
    'intent':'Continue actual source reconstruction and physical robot execution; never weaken acceptance checks.'},indent=2))
print(CONT/'grasp.yaml')

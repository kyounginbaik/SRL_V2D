"""Run SPIDER's own OakInk processor on the whole write_on_whiteboard segment. Only the task lookup pickle and a one-segment program file are synthesised; pre-grasp 0 and post-grasp 11.375 s selects exactly frames 2109-3474 at the stock 50 Hz."""
import numpy as _n
for _a,_t in dict(bool=bool,int=int,float=float,complex=complex,object=object,str=str,unicode=str,nan=float("nan"),inf=float("inf")).items():
    if _a not in _n.__dict__: setattr(_n,_a,_t)
import json,pickle
W='/scratch2/exaFLOPs/spider_wide/ds'; H='/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write'; T='write_wb'
TOK='scene_02__A002++seq__e9aab4c73ce9d2c16b85__2023-04-22-09-37-57'; OBJ='O02@0029@00011'
P=json.load(open(f'/lustre/exaflops/retarget_bench/data/oakink2_root/program/program_info/{TOK}.json'))
json.dump({k:v for k,v in P.items() if v['primitive']=='write_on_whiteboard'},open(f'{H}/prefix/program/program_info/{TOK}.json','w'))
pickle.dump({'right':dict(data_path=[f"x/{TOK}.pkl"],obj_id=[OBJ],obj_mesh_path=[f"x/align_ds/{OBJ}/model.obj"])},open(f'{W}/raw/oakink/{T}_bimanual.pkl','wb'))
from spider.process_datasets.oakinkv2 import main
main(dataset_dir=W,oakink2_prefix=f'{H}/prefix',embodiment_type='right',task=T,pre_grasp_seconds=0.0,post_grasp_seconds=11.375,mano_assets_root='/scratch2/exaFLOPs/daid_oakink/mano_root',show_viewer=False)
print('TASK',T)

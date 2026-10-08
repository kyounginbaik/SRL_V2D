import csv
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from score import reference_path, object_chamfer_cm, object_seed
from v2dlb.mesh_budget import budget_mesh
out=Path(sys.argv[1])
ids=pq.read_table('/data1/video_to_data_challenge/v2d_submission_kit/data/track_3_mesh_sample_submission.parquet',columns=['row_id']).column(0).to_pylist()
prefix=ids[0].rsplit('_',2)[0]+'_'
nv=sum(x.startswith(prefix+'0_') for x in ids)
nf=sum(x.startswith(prefix+'1_') for x in ids)
assert nv>0 and nf>0
candidate=budget_mesh(out/'generated.glb',nf,nv)
ref=budget_mesh(reference_path('water_pitcher'),nf,nv)
rows=[]
with (ROOT.parent/'cd_o_per_episode_object.csv').open() as f:
 for row in csv.DictReader(f):
  if row['object_name']!='water_pitcher': continue
  ep,slot=int(row['episode']),int(row['object_slot'])
  value=object_chamfer_cm(ref,candidate,object_seed(ep,slot))
  rows.append({'episode':ep,'slot':slot,'cd_o_cm':value})
  print(rows[-1],flush=True)
result={'mesh':str(out/'generated.glb'),'reference_used_only_for_scoring':True,'packed':True,'vertex_budget':nv,'face_budget':nf,'scores':rows,'mean_cd_o_cm':float(np.mean([x['cd_o_cm'] for x in rows]))}
(out/'scores.json').write_text(json.dumps(result,indent=2))

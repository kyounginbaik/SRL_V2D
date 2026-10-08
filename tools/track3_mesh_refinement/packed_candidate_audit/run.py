"""Re-rank existing video-derived candidates under the actual submission mesh budget."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from score import reference_path, object_chamfer_cm, object_seed
from v2dlb.mesh_budget import budget_mesh

def main():
 with (ROOT.parent/'candidate_scores.csv').open() as f:
  candidates=list(csv.DictReader(f))
 names=sorted({x['object_name'] for x in candidates})
 name=names[int(os.environ['SLURM_ARRAY_TASK_ID'])]
 with (ROOT.parent/'cd_o_per_episode_object.csv').open() as f:
  roster=[r for r in csv.DictReader(f) if r['object_name']==name]
 ids=pq.read_table('/data1/video_to_data_challenge/v2d_submission_kit/data/track_3_mesh_sample_submission.parquet',columns=['row_id']).column(0).to_pylist()
 prefix=ids[0].rsplit('_',2)[0]+'_'
 nv=sum(x.startswith(prefix+'0_') for x in ids)
 nf=sum(x.startswith(prefix+'1_') for x in ids)
 assert nv>0 and nf>0
 reference=budget_mesh(reference_path(name),nf,nv)
 output=ROOT/(name+'.json')
 results=json.loads(output.read_text()) if output.exists() else []
 seen={(r['mesh'],r['sha256']) for r in results}
 for item in candidates:
  if item['object_name']!=name: continue
  path=Path(item['mesh'])
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if (str(path),digest) in seen: continue
  candidate=budget_mesh(path,nf,nv)
  scores=[]
  for row in roster:
   ep,slot=int(row['episode']),int(row['object_slot'])
   score=object_chamfer_cm(reference,candidate,object_seed(ep,slot))
   scores.append({'episode':ep,'slot':slot,'cd_o_cm':score})
  result={'object_name':name,'mesh':str(path),'sha256':digest,'packed_mean_cd_o_cm':float(np.mean([s['cd_o_cm'] for s in scores])), 'scores':scores,'vertex_budget':nv,'face_budget':nf,'scope':'public in-sample candidate comparison; not leaderboard score'}
  results.append(result)
  tmp=output.with_suffix('.tmp'); tmp.write_text(json.dumps(results,indent=2)); tmp.replace(output)
  print(json.dumps({k:v for k,v in result.items() if k!='scores'}),flush=True)
 (ROOT/(name+'.completed.json')).write_text(json.dumps({'candidates':len(results),'best':min(results,key=lambda x:x['packed_mean_cd_o_cm'])},indent=2))

if __name__=='__main__': main()

"""Materialize a separate Eval candidate from completed public selection experiments."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/data1/SRL_V2D/challenge_submissions/track3/eval_mesh_refined_20261008')
OLD=BASE.parent/'eval_submit_20261008'

def main():
 selection=json.loads((ROOT/'combined_selection.json').read_text())
 old=json.loads((OLD/'mesh_manifest.json').read_text())
 manifest=[]
 for entry in old:
  name=entry['object_name']
  chosen=selection['objects'].get(name)
  source=Path(chosen['mesh']) if chosen else Path(entry['source'])
  expected=chosen['sha256'] if chosen else entry['source_sha256']
  digest=hashlib.sha256(source.read_bytes()).hexdigest()
  if digest!=expected: raise ValueError(f'Source changed: {source}')
  target=BASE/'meshes'/f"episode_{entry['episode']:06d}"/f'{name}.glb'
  target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(source,target)
  row={**entry,'mesh':str(target),'source':str(source),'source_sha256':digest,'source_method':chosen['method'] if chosen else entry['source_method']}
  if chosen: row['public_packed_cd_o_cm']=chosen['score_cm']
  manifest.append(row)
 if len(manifest)!=36: raise ValueError('Unexpected roster')
 (BASE/'mesh_manifest.json').write_text(json.dumps(manifest,indent=2))
 shutil.copy2(ROOT/'combined_selection.json',BASE/'public_selection.json')
 (BASE/'status.json').write_text(json.dumps({'meshes_prepared':36,'public_packed_mean_cm':selection['mean_cd_o_cm'],'leaderboard_score':None,'packed_submission_ready':False,'remaining':['Record and publish producing code commit','Pack with official submission tool','Audit packed meshes','Obtain official leaderboard result'],'original_candidate_preserved':str(OLD)},indent=2))
 print(f'Prepared {len(manifest)} candidate meshes at {BASE}; packing and leaderboard verification remain.')

if __name__=='__main__': main()

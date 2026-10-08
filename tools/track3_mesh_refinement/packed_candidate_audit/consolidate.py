"""Select among completed public experiments; keep the previous submission untouched."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BENCH=ROOT.parent

def main():
 originals=json.loads((BENCH/'selected_video_only/selection_manifest.json').read_text())['objects']
 choices={name:[] for name in originals}
 for name in choices:
  path=ROOT/(name+'.completed.json')
  if not path.exists(): raise RuntimeError(f'Unfinished packed audit: {name}')
  for item in json.loads((ROOT/(name+'.json')).read_text()):
   choices[name].append({'mesh':item['mesh'],'score_cm':item['packed_mean_cd_o_cm'],'method':'existing video-derived candidate','pairs':len(item['scores'])})
  path=BENCH/'smoothing_ablation/packed'/name/'completed.json'
  item=json.loads(path.read_text())
  for variant,value in item['means_cm'].items():
   if variant=='original': continue
   choices[name].append({'mesh':str(path.parent/(variant+'.glb')),'score_cm':value,'method':'Taubin-smoothed video-derived candidate','pairs':item['pairs']})
 for folder in ['dish_rack_additional_views','rack_geometric']:
  expected=6 if folder=='dish_rack_additional_views' else 3
  paths=list((BENCH/folder).glob('*/scores.json'))
  if len(paths)!=expected: raise RuntimeError(f'Unfinished {folder}: {len(paths)}/{expected}')
  for path in paths:
   item=json.loads(path.read_text())
   choices['plastic_dish_rack'].append({'mesh':item['mesh'],'score_cm':item['mean_cd_o_cm'],'method':folder,'pairs':len(item['scores'])})
 winners={}
 for name,items in choices.items():
  winner=dict(min(items,key=lambda x:x['score_cm']))
  winner['sha256']=hashlib.sha256(Path(winner['mesh']).read_bytes()).hexdigest()
  winner['candidates_compared']=len(items)
  winners[name]=winner
 pairs=sum(w['pairs'] for w in winners.values())
 result={'scope':'Public in-sample candidate selection at official mesh budget; not a hidden leaderboard score','pairs':pairs,'mean_cd_o_cm':sum(w['pairs']*w['score_cm'] for w in winners.values())/pairs,'objects':winners,'baseline_packed_cm':json.loads((BENCH/'smoothing_ablation/packed/summary.json').read_text())['packed_baseline_cm']}
 (ROOT/'combined_selection.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))

if __name__=='__main__': main()

from pathlib import Path
import json,sys
import wandb
b=Path(__file__).resolve().parents[1];phase=sys.argv[1] if len(sys.argv)>1 else 'pilot';seeds=[int(x) for x in sys.argv[2:]] or [43,44,45]
api=wandb.Api(timeout=45);rows=[]
for seed in seeds:
 rid=f'uniretarget-wrist-eigend-mixed-500k-ep01-20261013-{phase}-episode_000001-chunk-s{seed}'
 run=api.run(f'DexDreamer/v2d-track3-uniretarget/{rid}')
 summary=dict(run.summary);row={'seed':seed,'id':rid,'url':run.url,'state':run.state,'name':run.name,'environment_steps':summary.get('environment_steps'),'voc_scale':summary.get('curriculum/voc_scale'),'reward_hand_keypoints':run.config.get('hand_keypoint_reward_weight'),'reward_fingers':run.config.get('finger_joint_tracking_reward_weight'),'palm_relative_weight':run.config.get('keypoint_palm_relative_weight'),'videos':{k:v for k,v in summary.items() if k.startswith('videos/')}}
 rows.append(row)
(b/f'wandb_remote_verification_{phase}.json').write_text(json.dumps(rows,indent=2,default=lambda value:dict(value)));print(json.dumps(rows,indent=2,default=lambda value:dict(value)))

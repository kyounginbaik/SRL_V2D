from pathlib import Path
import numpy as np,json
b=Path(__file__).resolve().parents[1];root=b/'runs/smoke/episode_000001/chunk/seed_43/training';ref=np.load(root/'references/reference_0000.npz');batch=np.load(root/'search_batches/step_65536/proposals.npz');q=batch['joint_q'];actions=batch['actions'];base=ref['joint_q']
assert actions.shape==(2048,204);assert len(np.unique(actions,axis=0))==2048
fingers={}
for side in ['left','right']:
 f=ref[side+'/finger_joint_pos'];starts=[i for i in range(base.shape[1]-21) if np.array_equal(base[:,i:i+22],f)];assert len(starts)==1
 start=starts[0];assert np.all(q[:,:,start:start+22]==f[None]);fingers[side]={'q_start':start,'max_difference_rad':0.}
assert np.all(q[:,[0,-1]]==base[None,[0,-1]])
update=json.loads((root/'upper_updates.jsonl').read_text().splitlines()[-1]);report={'passed':True,'candidate_count':len(actions),'distinct_candidates':2048,'action_dimensions':204,'finger_invariance':fingers,'endpoints_fixed':True,'full_batch_seconds':update['seconds'],'accepted':update['accepted'],'task_success_claimed':False}
(b/'search_evidence_validation.json').write_text(json.dumps(report,indent=2));print(report)

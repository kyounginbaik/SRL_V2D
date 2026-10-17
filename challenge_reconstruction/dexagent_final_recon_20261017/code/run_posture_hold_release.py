from pathlib import Path
import json
import numpy as np
from search_posture_policy_v4 import Search,SCENE,B
from evaluate_posture_hold import main as evaluate
from audit_episode3_grip_quality import main as audit


def main():
    out=B/'refinement_20261018/learned_episode3_hold_release';out.mkdir(exist_ok=True)
    sources={'anatomy':B/'refinement_20261018/learned_episode3_posture_anatomy/policy.npz'}
    run=B/'refinement_20261018/learned_episode3_posture_grasp'
    progress=json.loads((run/'progress.json').read_text());search=Search();search.configure(progress['best_parameters'])
    target=out/'grasp';target.mkdir(exist_ok=True);sources['grasp']=target/'source_policy.npz';np.savez_compressed(sources['grasp'],**search.policy.data)
    (target/'source_search_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    for name,source in sources.items():
        target=out/name;target.mkdir(exist_ok=True)
        if (target/'evaluation/result.json').exists():raise FileExistsError(target)
        with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        data['posture_hold_phase_s']=14.2;checkpoint=target/'policy.npz';np.savez_compressed(checkpoint,**data)
        evaluate(3,str(SCENE),str(checkpoint),str(target/'evaluation'))
        audit(target/'evaluation',sample_stride=5)


if __name__=='__main__':main()

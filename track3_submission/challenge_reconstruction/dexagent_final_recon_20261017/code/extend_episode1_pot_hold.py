"""Retain the acquired pot grasp while the right hand replaces the lid."""
import json
import numpy as np
from repair_episode1_registration import OUT
from physics_pickup_clearance_gravity import ClearanceExperiment,dump,execute_reference,evaluate_files,TR

def run():
    root=OUT/'patch_height094';source=root/'pot_cem_v1';out=root/'pot_holdfull';out.mkdir(exist_ok=True)
    prev=json.loads((source/'result.json').read_text());spec=dict(prev['spec']);spec['times']=dict(spec['times'],release=17.0)
    exp=ClearanceExperiment(spec);Q,_=exp.commands(np.asarray(prev['parameters']));old=TR.load_reference(source/'reference.npz',exp.m)['qpos']
    assert np.array_equal(Q[:221],old[:221])
    untouched=np.setdiff1d(np.arange(exp.m.nq),exp.cols);assert np.array_equal(Q[:,untouched],old[:,untouched])
    ref=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Retain physically acquired left pot grasp through lid replacement, original object clock and targets',arms_coupled=False,floating_runtime=True)
    execution=execute_reference(spec['scene'],ref,out/'execution');metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    result=dict(reference=str(ref),execution=execution,metrics=metrics,spec=spec,parameters=prev['parameters'],
        change='Keep same acquired left grasp rather than returning to failed original finger posture at11s',preserved_prefix_seconds=11.,unchanged_other_robot_and_object_columns=True)
    dump(out/'result.json',result);print(json.dumps(metrics['official_style_reference_proxy'],indent=2),flush=True)

if __name__=='__main__':run()

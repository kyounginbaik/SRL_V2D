"""Keep immutable evaluated policy/reference pairs and a monotone best-demo pointer."""
from copy import deepcopy
import json
import numpy as np
from flash_chord.retargeting.fixed_start import historical_prefix_check, better_demo


class DemoArchive:
    def __init__(self,editor):
        self.editor=editor
        self.directory=editor.output.parent/'demo_archive'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.position=self.orientation=self.cost=self.best=None

    def check(self,p,a):
        return historical_prefix_check(p,a,self.position,self.orientation,self.editor.baseline.fps,
                                       **self.editor.timing_options)

    def consider(self,binding,learner,p,a,metrics,cost,step,origin,*,save_pair=False):
        check=self.check(p,a)
        promoted=better_demo(check,cost,self.cost)
        pair=None
        if promoted or save_pair:
            pair=self._save_pair(binding,learner,p,a,metrics,cost,step,origin,check)
        if promoted:
            self.position,self.orientation=np.array(p,copy=True),np.array(a,copy=True)
            self.cost=float(cost);self.best=pair
            path=self.directory.parent/'best_demo.json';temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(pair,indent=2,allow_nan=False)+'\n')
            temporary.replace(path)
        report=dict(step=int(step),origin=origin,
            current_first_failure_seconds=check['current_success_prefix_frames']/self.editor.baseline.fps,
            previous_best_prefix_seconds=check['saved_prefix_frames']/self.editor.baseline.fps,
            best_success_prefix_seconds=self.best['success_prefix_seconds'] if self.best else 0.,
            current_prefix_preserved=check['saved_prefix_preserved'],
            rejection_reasons=check['saved_prefix_rejection_reasons'],
            best_pair_updated=promoted,full_horizon_cost=float(cost),
            best_full_horizon_cost=self.cost,initial_state_sha256=self.editor.initial_state_sha256,
            reference_version=self.editor.version,
            pair_directory=None if pair is None else pair['directory'])
        with (self.directory.parent/'progress_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(report,allow_nan=False)+'\n')
        return report

    def _save_pair(self,binding,learner,p,a,metrics,cost,step,origin,check):
        from flash_chord.retargeting.reference_snapshot import save_snapshot
        from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
        from flash_chord.training.flash_sac.checkpoint import actor_inference_state,save_actor_checkpoint
        editor=self.editor
        if editor.config.get('training', {}).get('curriculum') is not None:
            if not {'curriculum_stage', 'curriculum_stage_json'} <= editor.metadata.keys():
                raise RuntimeError('Archive requires the actual curriculum stage from the training runner')
        directory=self.directory/f'step_{step:012d}_{origin}'
        directory.mkdir(parents=True,exist_ok=False)
        reference=directory/'reference.npz'
        digest=save_snapshot(reference,binding,editor.projector.layout,
            provenance=dict(step=step,origin=origin,version=editor.version,
                            initial_state_sha256=editor.initial_state_sha256))
        config=deepcopy(editor.config)
        config['task']['robot_reference_path']=str(reference)
        config['task']['robot_reference_sha256']=digest
        metadata=dict(editor.metadata)
        metadata.update({CONFIG_METADATA_KEY:json.dumps(config,sort_keys=True),
                         'environment_steps':int(step),'robot_reference_version':editor.version,
                         'initial_state_sha256':editor.initial_state_sha256})
        policy=directory/f'policy_{step}.safetensors'
        save_actor_checkpoint(policy,actor_inference_state(learner),metadata=metadata)
        np.savez_compressed(directory/'object_errors.npz',position_m=p,orientation_rad=a)
        (directory/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        record=dict(directory=str(directory),policy=str(policy),reference=str(reference),
            reference_sha256=digest,initial_state_sha256=editor.initial_state_sha256,
            object_command_sha256=editor.command_sha256,environment_steps=int(step),
            reference_version=editor.version,origin=origin,
            success_prefix_seconds=check['current_success_prefix_frames']/editor.baseline.fps,
            full_horizon_cost=float(cost),purpose='deployable pair; not a complete training resume snapshot')
        (directory/'pair.json').write_text(json.dumps(record,indent=2)+'\n')
        return record

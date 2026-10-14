"""Frozen-policy comparison of whole-trajectory and local full-hand search.

Each arm starts from one identical saved policy/reference. Search batches,
VOC comparisons, and paired perturbed confirmations are preserved on disk.
This diagnostic performs no lower-policy training.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import create_actor_template
from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint
from flash_chord.retargeting.parallel_editor import ParallelReferenceEditor
from flash_chord.retargeting.ttsa_editor import upper_cost
from flash_chord.retargeting.local_editor import LocalReferenceEditor

VARIANTS = {'old_chunk':(1.,'full',952),'local_u2':(1.,'full',56),'local_u32':(1.,'full',56),'local_onset_u2':(1.,'full',56),'local_onset_u32':(1.,'full',56),'local_final_u32':(1.,'full',56)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--worlds', type=int, default=2048)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--wandb-project', default='v2d-track3-uniretarget')
    p.add_argument('--wandb-entity', default='DexDreamer')
    p.add_argument('--no-wandb', action='store_true')
    a = p.parse_args()
    if (a.output/'upper_updates.jsonl').exists():
        raise RuntimeError('Use a fresh output directory; refusing to overwrite evidence')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'status.json').write_text(json.dumps({'stage':'initializing'}))
    (a.output/'gpu.json').write_text(json.dumps({'job':__import__('os').environ.get('SLURM_JOB_ID'),'worlds':a.worlds}))
    metadata = read_checkpoint_metadata(a.checkpoint)
    saved = checkpoint_config(metadata)
    chunk_seconds, scope, expected_dim = VARIANTS[a.variant]
    original_seed = saved['training']['seed']
    saved['scene']['world_count'] = a.worlds
    saved['training']['seed'] = a.seed
    saved['training']['resume'] = False
    cfg = OmegaConf.create(saved)
    training = instantiate_typed(cfg.training, TrainingConfig)
    ec = instantiate_typed(cfg.env, RLEnvConfig)
    emb = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    config = {'variant':a.variant, 'search_seed':a.seed, 'lower_seed':original_seed,
              'checkpoint':str(a.checkpoint), 'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'starting_reference':saved['task'].get('robot_reference_path'),
              'starting_reference_sha256':saved['task'].get('robot_reference_sha256'),
              'candidate_count':a.worlds, 'upper_action_dimension':expected_dim,
              'lower_policy_frozen':True, 'evaluation_voc':0., 'chunk_seconds':chunk_seconds,
              'scope':scope, 'rounds':a.rounds, 'boundary_seconds':.25, 'publication_blend':1. if a.variant.startswith('local') else .25,'upper_updates_per_batch':32 if a.variant.endswith('u32') else 2,
              'interpretation':'one-batch search diagnostic; does not establish long-term trainability'}
    (a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(entity=a.wandb_entity, project=a.wandb_project,
            name=f'local-refinement-probe-{a.variant}-s{a.seed}',
            group='local-refinement-frozen-policy-20261014', config=config, dir=str(a.output))
    try:
        with wp.ScopedDevice(training.device):
            setup = setup_scene(parquet=cfg.task.parquet, control_fps=ec.sim.fps,
                motion_speed=cfg.task.motion_speed, embodiment=emb, collision=collision,
                world_count=a.worlds, include_support=cfg.scene.support,
                decompose_objects=cfg.scene.decompose_objects,
                contact_friction=cfg.scene.get('contact_friction',1.),
                object_free_joint_damping=cfg.scene.get('object_free_joint_damping',0.),
                source_frame_playback=cfg.task.source_frame_playback,
                motion_start_frame=cfg.task.motion_start_frame, motion_end_frame=cfg.task.motion_end_frame,
                robot_reference_path=cfg.task.get('robot_reference_path'),
                robot_reference_sha256=cfg.task.get('robot_reference_sha256'))
            options = deepcopy(saved['bilevel'])
            options.update(study_variant='chunk', chunk_seconds=chunk_seconds, edit_scope=scope,
                upper_trials=a.worlds, eval_worlds=a.worlds, warmup=1, interval=1,
                guided_contact_proposal=False)
            options.update(upper_updates=32 if a.variant.endswith('u32') else 2,
                upper_initial_std=.5 if a.variant.startswith('local') else .15,
                keep_frozen_replay=True, robust_confirmation=False,
                include_noop=True)
            if 'onset' in a.variant or 'final' in a.variant:
                options.update(local_segment_schedule=[5,1,6],robust_confirmation=True)
            if 'final' in a.variant:
                options.update(defer_acceptance_gates_to_confirmation=True,confirmation_top_k=3)
            editor_class=LocalReferenceEditor if a.variant.startswith('local') else ParallelReferenceEditor
            editor = editor_class(options=options, setup=setup, embodiment=emb,
                collision=collision, training=training, environment_config=saved,
                checkpoint_metadata=dict(metadata), output_dir=a.output)
            dimension = editor.basis.shape[1]*editor.joint_basis.shape[0]
            assert dimension == expected_dim, (dimension, expected_dim)
            if a.variant.startswith('episode'):
                assert editor.basis.shape[1] == 1
                np.testing.assert_allclose(editor.basis[5:-5,0],1.)
            probe = editor.candidate_joint_q(np.full(dimension,.5,np.float32))
            np.testing.assert_array_equal(probe[[0,-1]],editor.baseline.joint_q[[0,-1]])
            finger_ids = np.concatenate([h.finger_q_ids for h in editor.projector.layout.hands])
            finger_change = np.max(np.abs(probe[:,finger_ids]-editor.baseline.joint_q[:,finger_ids]))
            assert (finger_change == 0) if scope == 'wrists_only' else (finger_change > 0)
            # A one-world environment supplies the unchanged starting reference.
            env = RLEnv(replace(editor.eval_setup.scene, robot_reference=editor.binding),
                        editor.eval_setup.reference, config=deepcopy(editor.eval_config))
            env.reset(); adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
            _, template = create_actor_template(training, adapter.observation_dim, adapter.action_dim)
            state, _ = load_actor_checkpoint(a.checkpoint, template)
            learner = SimpleNamespace(actor=state)
            (a.output/'status.json').write_text(json.dumps({'stage':'searching'}))
            history=[]
            for round_index in range(1,a.rounds+1):
                replacement=editor(adapter, learner, round_index, 0)
                if replacement is not None: adapter=replacement
                item=json.loads((a.output/'search_batches'/f'step_{round_index}'/'scores.json').read_text())
                item.pop('candidates',None);history.append(item)
                (a.output/'round_summary.json').write_text(json.dumps(history,indent=2))
                if run:run.log({'search/round':round_index,'search/accepted':int(item['accepted']),
                    'search/version':item['version_after'],'search/seconds':item['seconds']})
            voc_evidence={}
            for scale in [0.,.95,1.]:
                before,_,_=editor.rollout(editor.baseline,learner,a.seed,voc_probe_scale=scale)
                after,_,_=editor.rollout(editor.binding,learner,a.seed,voc_probe_scale=scale)
                voc_evidence[str(scale)]={'before':before,'after':after}
            (a.output/'voc_comparison.json').write_text(json.dumps(voc_evidence,indent=2))
            # Validate the cumulative published change on the same three perturbed resets.
            if editor.version:
                final=editor.binding;editor.binding=editor.baseline
                editor.options['robust_confirmation']=True
                passed,checks=LocalReferenceEditor.confirm_candidate(editor,final,learner,a.seed,None,None)
                editor.binding=final
                (a.output/'robust_confirmation.json').write_text(json.dumps(checks,indent=2))
            record = json.loads((a.output/'upper_updates.jsonl').read_text().splitlines()[-1])
            baseline_cost = upper_cost(record['baseline'], np.zeros_like(editor.baseline.joint_q),
                editor.baseline.fps, 1., robust=True, contact_proximity_weight=editor.contact_proximity_weight)
            candidates = record['candidates']
            report = {'variant':a.variant,'seed':a.seed,'dimensions':dimension,
                'accepted':record['accepted'],'accepted_total':editor.version,'rounds':a.rounds, 'gate_pass_count':sum(c['gate_passed'] for c in candidates),
                'improving_count':sum(c['cost']<baseline_cost for c in candidates),
                'baseline_cost':baseline_cost,'best_raw_cost':min(c['cost'] for c in candidates),
                'best_valid_cost':min([c['cost'] for c in candidates if c['gate_passed']],default=None),
                'seconds':record['seconds'],'baseline':record['baseline'],
                'confirmation':record.get('validation'), 'selected_trial':record['selected_trial'],
                'published_finger_rms_rad':float(np.sqrt(np.mean((editor.binding.joint_q[:,finger_ids]-editor.baseline.joint_q[:,finger_ids])**2)))}
            (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            if run:
                run.log({k:v for k,v in report.items() if isinstance(v,(int,float,bool))})
                run.summary['result'] = report
            (a.output/'status.json').write_text(json.dumps({'stage':'complete'}))
            print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','confirmation')},indent=2),flush=True)
    except Exception as error:
        (a.output/'status.json').write_text(json.dumps({'stage':'failed','error':str(error)}))
        if run: run.finish(exit_code=1)
        raise
    else:
        if run: run.finish()

if __name__ == '__main__':
    main()

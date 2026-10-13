"""Matched frozen-policy search: 12/56/204/952 parameters, no lower training.

An episode delta is one shared coefficient vector with the existing endpoint
ramp and joint/velocity projection. This is ReActor-inspired, not a reproduction.
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

VARIANTS = {'episode_wrist': (60., 'wrists_only', 12),
            'episode_full': (60., 'full', 56),
            'chunk_wrist': (1., 'wrists_only', 204),
            'chunk_full': (1., 'full', 952)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--worlds', type=int, default=2048)
    p.add_argument('--wandb-project', default='v2d-track3-uniretarget')
    p.add_argument('--wandb-entity', default='DexDreamer')
    p.add_argument('--no-wandb', action='store_true')
    a = p.parse_args()
    if (a.output/'upper_updates.jsonl').exists():
        raise RuntimeError('Use a fresh output directory; refusing to overwrite evidence')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'status.json').write_text(json.dumps({'stage':'initializing'}))
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
              'scope':scope, 'rounds':1, 'boundary_seconds':.25, 'publication_blend':.25,
              'interpretation':'one-batch search diagnostic; does not establish long-term trainability'}
    (a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(entity=a.wandb_entity, project=a.wandb_project,
            name=f'episode-delta-probe-{a.variant}-s{a.seed}',
            group='episode-delta-frozen-policy-20261013', config=config, dir=str(a.output))
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
            editor = ParallelReferenceEditor(options=options, setup=setup, embodiment=emb,
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
            editor(adapter, learner, 1, 0)
            record = json.loads((a.output/'upper_updates.jsonl').read_text().splitlines()[-1])
            baseline_cost = upper_cost(record['baseline'], np.zeros_like(editor.baseline.joint_q),
                editor.baseline.fps, 1., robust=True, contact_proximity_weight=editor.contact_proximity_weight)
            candidates = record['candidates']
            report = {'variant':a.variant,'seed':a.seed,'dimensions':dimension,
                'accepted':record['accepted'], 'gate_pass_count':sum(c['gate_passed'] for c in candidates),
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

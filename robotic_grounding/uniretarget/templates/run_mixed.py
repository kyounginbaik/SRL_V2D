"""Fresh episode-1 mixed-reference bilevel smoke/pilot, with checkpoint media."""
from pathlib import Path
import argparse,json,os,subprocess,sys,hashlib
B=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','pilot'],default='smoke');p.add_argument('--seed',type=int,default=42);p.add_argument('--worlds',type=int,default=None);p.add_argument('--dry-run',action='store_true');a=p.parse_args();a.worlds=a.worlds or (64 if a.phase=='smoke' else 2048)
 source_manifest=B.parent/'source_manifest.json'
 source_hashes=json.loads(source_manifest.read_text())
 for relative,checksum in source_hashes.items():
  assert hashlib.sha256((B/'flash_chord'/relative).read_bytes()).hexdigest()==checksum,relative
 f=B/'frozen/episode_000001';manifest=json.loads((B/'frozen_manifest.json').read_text())[0]
 for rel,h in {'reference.parquet':manifest['runtime_sha256'],**manifest['asset_sha256']}.items():
  assert hashlib.sha256((f/rel).read_bytes()).hexdigest()==h,rel
 run=B/'runs'/a.phase/'episode_000001/chunk'/f'seed_{a.seed}';train=run/'training';train.mkdir(parents=True,exist_ok=True)
 if list(train.glob('policy_*.safetensors')) or (train/'upper_updates.jsonl').exists():raise RuntimeError('Fresh runs only; complete bilevel resume is not implemented')
 total=65536 if a.phase=='smoke' else 100663296;interval=65536 if a.phase=='smoke' else 1003520
 search_interval=32768 if a.phase=='smoke' else 501760
 cmd=['scripts/train_flash_sac.py','embodiment=sharpa_hands_stronger','action.wrist_kp=100.0','action.wrist_tau_max=90.0',f'task.parquet={f/"reference.parquet"}','task.motion_speed=1.0',f'reset.seed={a.seed}',f'training.seed={a.seed}',f'scene.world_count={a.worlds}','scene.support=false','sim.njmax=2048','sim.nconmax=512','sim.substeps=20',f'training.total_environment_steps={total}','training.updates_per_collection=8','training.replay.capacity=1048576',f'training.replay.minimum_size={32768 if a.phase=="smoke" else 102400}','training.replay.batch_size=1024',f'training.checkpoint.save_interval_environment_steps={interval}','+adaptive_voc.mode=performance_gate',f'+adaptive_voc.interval={32768 if a.phase=="smoke" else 501760}','+adaptive_voc.required_passes=3','+adaptive_voc.reset_on_reference_change=false','+adaptive_voc.scales=[1.0,0.95,0.9,0.85,0.8,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.1,0.05,0.025,0.01,0.0]','env.voc_scale=1.0','reset.voc_decay_steps=0','+bilevel.engine=response_bilevel','+bilevel.edit_scope=full',f'+training.network.eigen_basis_path={B}/priors/sharpa_bimanual_residual_basis.npz',f'+training.network.eigen_basis_sha256={hashlib.sha256((B/"priors/sharpa_bimanual_residual_basis.npz").read_bytes()).hexdigest()}','+bilevel.mode=reactor_wrench','+bilevel.study_variant=chunk','+bilevel.chunk_seconds=1.0',f'+bilevel.warmup={search_interval}',f'+bilevel.interval={search_interval}',f'+bilevel.eval_worlds={a.worlds}',f'+bilevel.upper_trials={a.worlds}','+bilevel.contact_proximity_weight=0.0','+bilevel.cws_weight=5.0','+bilevel.cws_tolerance=0.1','+bilevel.select_best_trial=true','+bilevel.upper_updates=0','+bilevel.upper_initial_std=0.5','+bilevel.include_noop=true','+bilevel.robust_confirmation=false','+reference_guided_hands=true','objective.hand_keypoints.weight=2.0','objective.hand_joint_pos.weight=0.25','+objective.hand_keypoint_palm_weight=10.0','+bilevel.wrist_translation_offset_m=0.03','+bilevel.wrist_translation_correction_speed_m_s=0.10','training.mixed_precision=false','training.compute_dtype=float32','logging.mode=disabled',f'output_dir={train}',f'hydra.run.dir={train/"hydra"}']
 config={'phase':a.phase,'episode':manifest,'variant':'chunk','ablation_variant':'chord_convex_response_bilevel','temporal_chunk_seconds':1.0,'submission_meshes':True,'seed':a.seed,'voc_mode':'performance_gate','voc_initial_scale':1.0,'voc_scales':[1.0,0.95,0.9,0.85,0.8,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.1,0.05,0.025,0.01,0.0],'voc_probe_interval':32768 if a.phase=='smoke' else 501760,'voc_required_passes':3,'voc_reduction_rule':'both objects pass next-lower-scale full-horizon evaluation three consecutive checks','total_environment_steps':total,'upper_trials':a.worlds,'eval_worlds':a.worlds,'replicas_per_candidate':1,'worlds':a.worlds,'checkpoint_interval':interval,'upper_search_interval':search_interval,'upper_search_warmup':search_interval,'planned_upper_search_rounds':(total-1)//search_interval,'training_command':cmd,'code_root':str(B/'flash_chord'),'contact_proximity':False,'reference_guided':True,'wrist_translation_bound_m':.03,'hand_keypoint_reward_weight':2.0,'finger_joint_tracking_reward_weight':0.25,'keypoint_palm_relative_weight':10.0,'upper_edit_scope':'full','upper_updates_per_batch':0,'local_translation_step_m':0.02,'robust_confirmation_resets':0,'lower_response':'convex ridge mean-head adaptation','guarantee_scope':'per-stage lower surrogate residual bound; not global neural RL optimum','upper_action_dimension':56,'lower_action_dimension':56,'finger_exploration':'official Sharpa PCA, rank 9 per hand; learned diagonal plus low-rank tanh Gaussian','eigen_basis_sha256':hashlib.sha256((B/'priors/sharpa_bimanual_residual_basis.npz').read_bytes()).hexdigest(),'initial_finger_variance_fraction_eigen':0.5,'original_overlay_opacity':0.45,'overlay_style':'transparent original hands plus synchronized handle closeups','wrist_control':{'kp':100.0,'kd':12.0,'effort_limit':90.0,'linear_speed_limit_m_s':None,'angular_speed_limit_rad_s':None},'media_mode':'synchronous at every checkpoint using the same GPU allocation; training paused during renders','contact_labels':'new geometric estimates; immutable during training'}
 probe=B.parent/'probes/local_final_u32/seed_43/summary.json'
 if probe.exists():
  diagnostic=json.loads(probe.read_text())
  config['frozen_probe_accepted_rounds']=diagnostic['accepted_total']
 config['experiment_interpretation']='Full training experiment; increased edit magnitude and more upper updates do not establish improved task success.'
 config['window_selection']={'max_windows':4,'window_seconds':1.0,'shoulder_seconds':0.5,'failure_position_m':0.05,'failure_orientation_deg':20,'failure_persistence_seconds':0.15,'selection':'before error growth, before threshold failure, sampled earlier history','both_hands':True}
 cmd.extend(['reset.fixed_initial_state=true','reset.always_reset_to_first_frame=true',
             '+bilevel.fixed_start=true','+bilevel.failure_baseline_worlds=0'])
 config['fixed_initial_state']=True
 config['training_reset']='exact frame-zero robot/object poses and zero velocities; no finger opening'
 config['historical_prefix_protection']=True
 config['best_demo_selection']='longest saved successful prefix, then full-horizon pose/wrench cost'
 config['window_selection'].update(detector='fixed_start_nominal',baseline_worlds=1,
     parallel=False,reset_translation_m=0.,reset_finger_degrees=0.,
     selection='first sustained failure from the original episode start; earlier error growth and history')
 config['rejected_search_preserves_lower_replay']=True
 config['voc_rotation_inertia']='full inertia tensor'
 config['voc_reset_on_reference_change']=False
 config['controller_matched_to']='wrist-eigend-mixed-search500k-c2048-pot200g-lid100g-pilot-episode_000001-chunk-s44'
 config['source_manifest_sha256']=hashlib.sha256(source_manifest.read_bytes()).hexdigest()
 config['upper_updates_per_batch']=0
 config['upper_proposal_method']='Gaussian candidate population with solved lower response for every candidate'
 config['lower_surrogate']='regularized proximal approximation to the frozen-critic SAC actor loss'
 config['lower_response_error_tolerance']=1e-5
 config['guarantee_document']=str(B/'flash_chord/docs/response_bilevel.txt')
 config['robust_confirmation']='disabled; original demonstration start, paired adapted no-edit control'
 config['acceptance_mode']='local_nominal'
 config['acceptance_lookahead_seconds']=1.0
 config['acceptance_position_regression_m']=0.05
 config['acceptance_orientation_regression_deg']=20.0
 config['acceptance_score']='both-object pose and wrench cost from error onset through failure+1s; prefix protected; large whole-episode/tail regressions rejected'
 cmd.extend(['+bilevel.acceptance_mode=local_nominal','+bilevel.acceptance_lookahead_seconds=1.0',
             '+bilevel.acceptance_position_regression_m=0.05','+bilevel.acceptance_orientation_regression_deg=20.0'])
 config['upper_contact_objective']={'type':'CHORD-style normalized squared tolerance-band support loss','tolerance':0.1,'weight':5.0,'missed_weight':1.0,'unintended_weight':1.0,'contact_regression_tolerance':None,'location_penalty':False,'location_gate':False}
 config['experiment_interpretation']='Candidate-specific convex policy-head responses with CHORD-style upper contact guidance; no full neural-RL convergence guarantee.'
 (run/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
 if a.dry_run:print(json.dumps(config,indent=2));return
 def stage(s,**kw):(run/'stage.json').write_text(json.dumps({'stage':s,**kw},indent=2)+'\n')
 env=os.environ.copy();env['V2D_CHECK_FINITE']='1';env['UNIRETARGET_CHECKPOINT_MEDIA']=str(B/'code/checkpoint_media.py');env['UNIRETARGET_PHASE']=a.phase;env['UNIRETARGET_SEED']=str(a.seed)
 try:
  stage('training');subprocess.run([sys.executable,*cmd],cwd='/workspace',env=env,check=True)
  checkpoints=sorted(train.glob('policy_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]));assert checkpoints
  stage('unassisted_evaluation');out=run/'evaluation';out.mkdir(exist_ok=True)
  subprocess.run([sys.executable,'scripts/evaluate_policy.py',f'evaluation.checkpoint={checkpoints[-1]}','evaluation.world_count=1','evaluation.reset_mode=explicit',f'evaluation.metrics_output={out/"metrics.json"}',f'hydra.run.dir={out/"hydra"}','logging.mode=disabled'],cwd='/workspace',check=True)
  assert not (train/'nonfinite_transition.json').exists()
  updates=[json.loads(line) for line in (train/'upper_updates.jsonl').read_text().splitlines()]
  assert updates, 'Upper search never executed'
  media=list(run.glob('checkpoint_videos/step_*/manifest.json'));assert len(media)==len(checkpoints),'Missing evaluation videos'
  assert len(list(run.glob('voc_checkpoint_videos/step_*/manifest.json')))==len(checkpoints),'Missing current-VOC videos'
  assert list(run.glob('reference_overlays/version_*/transparent_closeups/complete.json')),'Missing reference overlay'
  stage('complete')
 except Exception as e:stage('failed',error=str(e));raise
 finally:
  subprocess.run([sys.executable,str(B/'code/wandb_sync.py'),a.phase,'episode_000001','chunk',str(a.seed),'--once'],cwd='/workspace')
if __name__=='__main__':main()

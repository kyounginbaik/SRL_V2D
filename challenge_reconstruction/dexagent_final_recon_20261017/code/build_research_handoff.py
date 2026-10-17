#!/usr/bin/env python3
"""Freeze an explicit, audited five-episode research selection; never a submission."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json,hashlib
import numpy as np
import mujoco
from summarize_refinement_audits import metric_summary, grasp_summary

B=Path(__file__).resolve().parents[1]
R=B/'refinement_20261017'
DEFAULTS={n:R/'block_polish'/f'episode_{n:06d}' for n in (1,3,5,8,10)}
DEFAULTS[8]=R/'feedback_polish/episode_000008/both_0.5'
REASONS={1:'Audited core candidate; intended pot/lid grasps remain unresolved.',
 3:'Retained pot grasp with low palm-relative drift; spoon pickup remains unresolved.',
 5:'Both objects picked up and returned; orientation and retention errors remain. Alternatives are listed separately.',
 8:'Independently verified proxy pass; feedback improves cup tracking, although pot peak lift remains underestimated.',
 10:'Audited core candidate; blue-cup pickup and nesting remain unresolved.'}

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def artifact(path):
 p=Path(path).resolve()
 if not p.is_file():raise FileNotFoundError(p)
 return dict(path=str(p),workspace_relative=str(p.relative_to(B)) if p.is_relative_to(B) else None,sha256=digest(p),bytes=p.stat().st_size)

def selected_entry(number,directory,reason):
 ep=f'episode_{number:06d}';directory=directory.resolve();inp=B/'episodes'/ep
 rp=directory/'result.json' if (directory/'result.json').exists() else directory/'report.json';r=read(rp)
 if r['episode']!=ep:raise ValueError('Selected episode mismatch')
 ex=r['execution'];metric=read(directory/'audit/mp_sr_independent.json');grasp=read(directory/'audit/grasp_stability_frames.json')
 if not all([ex['physics_verified'],ex['execution_completed'],ex['independent_control_replay']['ok'],metric['full_horizon']]):raise ValueError('Full physics/replay/metric validation is required')
 if r.get('learned_policy',False):raise ValueError('This research handoff deliberately excludes unverified policy/checkpoint claims')
 ref=Path(r['reference']);original=inp/'tracking_reference.npz';model=mujoco.MjModel.from_xml_path(ex['scene'])
 with np.load(ref,allow_pickle=False) as z, np.load(original,allow_pickle=False) as raw:
  names=z['joint_names'].astype(str).tolist();source_names=raw['joint_names'].astype(str).tolist()
  if set(names)!=set(source_names) or len(names)!=len(set(names)):raise ValueError('Reference coordinate schema mismatch')
  q=z['qpos'][:,[names.index(k) for k in source_names]];initial=raw['qpos'];objcols=[i for i,n in enumerate(source_names) if not n.startswith('robot_')]
  if q.shape!=initial.shape or float(z['fps'])!=float(raw['fps']):raise ValueError('Reference timeline changed')
  if not np.array_equal(q[:,objcols],initial[:,objcols]):raise ValueError('Object reference targets changed')
  expected=np.arange(len(q))/float(z['fps'])
 with np.load(ex['trajectory'],allow_pickle=False) as z:
  if str(z['scene_sha256'])!=digest(ex['scene']):raise ValueError('Frozen scene hash mismatch')
  if len(z['qpos'])!=len(q) or not np.allclose(z['time_s'],expected,rtol=0,atol=.0021):raise ValueError('Recording duration mismatch')
  if not np.array_equal(z['initial_qpos'],model.key_qpos[0] if model.nkey else model.qpos0):raise ValueError('Initial state differs from frozen scene home')
  controls=dict(actuator_names=z['actuator_names'].astype(str).tolist(),
   frame_control_shape=list(z['ctrl'].shape),substep_control_shape=list(z['substep_ctrl'].shape),
   substep_timestep_seconds=float(model.opt.timestep),control_reference_fps=float(z['fps']),
   initial_state_is_recorded=True,control_replay_is_independently_verified=True)
 manifest=read(inp/'input_manifest.json')
 if digest(Path(manifest['source'])/'object_poses.npz')!=manifest['source_object_poses_sha256']:
  raise ValueError('Original object pose file changed since preparation')
 if digest(manifest['source_reference'])!=manifest['source_reference_sha256']:
  raise ValueError('Original source reference changed since preparation')
 for obj in manifest['objects']:
  if digest(obj['source_urdf'])!=obj['source_urdf_sha256']:
   raise ValueError('Original source URDF changed since preparation')
  for mesh in obj['meshes']:
   if digest(mesh['source'])!=mesh['sha256']:
    raise ValueError('Original source mesh changed since preparation')
 files={'result':artifact(rp),'robot_reference':artifact(ref),
  'execution_report':artifact(Path(ex['trajectory']).parent/'execution_report.json'),
  'scene':artifact(ex['scene']),'trajectory_with_controls':artifact(ex['trajectory']),
  'independent_metrics':artifact(directory/'audit/mp_sr_independent.json'),
  'grasp_diagnostic':artifact(directory/'audit/grasp_stability_frames.json'),
  'aligned_video':artifact(directory/'videos/reference_baseline_refined.mp4'),
  'source_manifest':artifact(inp/'input_manifest.json'),
  'metric_geometry':artifact(inp/'metric_geometry.json'),
  'original_object_poses':artifact(Path(manifest['source'])/'object_poses.npz')}
 ck=directory/'audit/search_replay_consistency.json'
 if ck.exists():
  if not read(ck)['consistent']:raise ValueError('Search/final execution mismatch')
  files['search_final_consistency']=artifact(ck)
 return dict(episode=ep,selection_reason=reason,selected_run=str(directory),
  proxy_metrics=metric_summary(metric),grasp_diagnostic=grasp_summary(grasp),
  original_object_targets_unchanged=True,original_timestamps_unchanged=True,full_physics_replay=True,
  original_reference_object_pose_urdf_mesh_hashes_unchanged=True,
  controls=controls,files=files,learned_policy=False,submission_checkpoint=None,submission_ready=False)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--select',action='append',default=[],help='Explicit override, e.g. 5=/absolute/completed/run')
 p.add_argument('--reason',action='append',default=[],help='Explicit override reason, e.g. 5=Retained both grasps better')
 p.add_argument('--output',type=Path,default=R/'research_handoff')
 a=p.parse_args();selected=dict(DEFAULTS);reasons=dict(REASONS)
 for value in a.select:
  key,path=value.split('=',1);key=int(key)
  if key not in selected:raise ValueError('Only episodes 1,3,5,8,10 are authorized in this handoff')
  selected[key]=Path(path)
 for value in a.reason:
  key,reason=value.split('=',1);reasons[int(key)]=reason
 entries=[selected_entry(n,selected[n],reasons[n]) for n in selected]
 report=dict(schema='dexagent.research_handoff.v1',generated_utc=datetime.now(timezone.utc).isoformat(),
  workspace=str(B),coverage_episodes=[e['episode'] for e in entries],challenge_coverage='5 of 20; intentionally partial',
  reconstructed_reference_proxy_pass_count=sum(e['proxy_metrics']['mp_sr']==1 for e in entries),
  official_leaderboard_score=None,learned_policy=False,submission_ready=False,submission_checkpoint=None,
  has_learned_policy_checkpoint=False,submitted_checkpoint_status=False,
  submitted=False,selection_is_explicit=True,selection_uses_cost_alone=False,
  scope='Research references and measured physics recordings for follow-on refinement or policy training, not a challenge submission.',
  method_scope='Floating-Sharpa runtime adapted from DexAgent, followed by custom CEM/block reference optimization and robot-command feedback diagnostics; not certification of the native full DexAgent pipeline.',
  path_portability='Scene XML assets use absolute /data1/SRL_V2D paths. Preserve the directory root or remap and revalidate all asset references on another server.',
  assumptions=['Reconstructed object poses are proxy GT; confidential official GT is unavailable.',
   'Contact diagnostics do not prove force closure. Metric passing does not guarantee every motion peak is reproduced.',
   'A genuine trained policy, compatible official evaluation, valid provenance and complete required coverage remain necessary for submission.'],
  episodes=entries)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'selected_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 lines=['# Five-episode research handoff','',f"Selected proxy passes: **{report['reconstructed_reference_proxy_pass_count']}/5**. This is not a submission or learned-policy checkpoint package.",'',
  'The manifest freezes each selected reference, measured physics trajectory and controls, scene, source object poses, independent metrics, contact audit and aligned video by SHA-256. Selection is explicit; no per-object mixing of different rollouts is performed.','',
  '| Episode | Selected run | Proxy pass | Video |','|---|---|:---:|---|']
 for e in entries:
  lines.append(f"| {e['episode']} | {e['selected_run']} | {'yes' if e['proxy_metrics']['mp_sr'] else 'no'} | [video]({e['files']['aligned_video']['path']}) |")
 lines+=['','Use the complete recorded substep controls and exact initial state for deterministic control replay. A nominal reference alone is insufficient to reproduce a state-feedback controller. The scene uses absolute `/data1/SRL_V2D` asset paths; relocating it requires path remapping and revalidation.','',
  'Refresh with `code/build_research_handoff.py`. To change a selection, pass `--select EPISODE=/absolute/run --reason EPISODE="reason"`; the script fails if required independent audits, videos, original object targets, timeline, or replay checks are missing.','',
  'No selected learned-policy checkpoint exists. Official hidden-GT evaluation, genuine policy provenance and full submission coverage remain outstanding.']
 (a.output/'README.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(manifest=str(a.output/'selected_manifest.json'),proxy_pass_count=report['reconstructed_reference_proxy_pass_count'],submission_ready=False)))

if __name__=='__main__':main()

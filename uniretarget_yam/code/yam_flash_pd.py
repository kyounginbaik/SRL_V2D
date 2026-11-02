"""Arm B and the fixed-reference floor: a reference on YAM in Newton through the canonical evaluator, zero residual, no policy."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,warp as wp
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
import flash_chord
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.retargeting.evaluation_task import apply_task_protocol
from flash_chord.retargeting.physical_reference_evaluator import official_scores
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
p=argparse.ArgumentParser();p.add_argument('--parquet',type=Path,required=True);p.add_argument('--urdf',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replicas',type=int,default=4);p.add_argument('--device',default='cpu');a=p.parse_args()
with initialize_config_dir(str(Path(flash_chord.__file__).parent/'configs'),version_base='1.3'):
    cfg=compose('train_flash_sac',overrides=['experiment=yam_sharpa_flash_sac',f"task.parquet='{a.parquet}'",f"embodiment.config.urdf_path._args_=['{a.urdf}']"])
cfg=OmegaConf.to_container(cfg,resolve=True)
manifest=dict(checkpoint=None,reference=None,reference_sha256=None,evaluation_task_protocol=dict(parquet=str(a.parquet),parquet_sha256=hashlib.sha256(a.parquet.read_bytes()).hexdigest(),collision=dict(_target_='flash_chord.scene.collision.CollisionPolicy',robot_scope='all',hand_self_collision=True,robot_object_collision=True,robot_support_collision=True,ground=True)))
r=apply_task_protocol(cfg,manifest);apply_evaluation_composition(r,world_count=a.replicas);config=instantiate_typed(OmegaConf.create(r['env']),RLEnvConfig);emb=instantiate_typed(OmegaConf.create(r['embodiment']),Embodiment);col=instantiate_typed(OmegaConf.create(r['collision']),CollisionPolicy)
with wp.ScopedDevice(a.device):
    t,s=r['task'],r['scene'];W=a.replicas;setup=setup_scene(parquet=t['parquet'],control_fps=config.sim.fps,motion_speed=t['motion_speed'],embodiment=emb,collision=col,world_count=W,include_support=s['support'],decompose_objects=s['decompose_objects'],contact_friction=s.get('contact_friction',1.),object_free_joint_damping=s.get('object_free_joint_damping',0.),robot_reference_path=None,robot_reference_sha256=None);env=RLEnv(setup.scene,setup.reference,config=config);env.reset(frame_id=0);F=env.scene.robot_reference.num_frames;print(json.dumps(dict(built=True,frames=F,worlds=W,nu=env.solver.mj_model.nu,ngeom=env.solver.mj_model.ngeom,action_dim=env.action.action_dim)),flush=True)
    if a.device.startswith('cuda'):env.capture_step();env.reset(frame_id=0)
    action=wp.zeros(W*env.action.action_dim,dtype=wp.float32,device=a.device);poses=[];qs=[];t0=time.monotonic();nq=env.scene.robot_reference.num_joint_q;qref=np.asarray(env.scene.robot_reference.joint_q,np.float32);names=list(env.scene.layout.scalar_joints.names);qid=list(env.scene.layout.scalar_joints.q_ids)
    for f in range(F-1):
        env.step(action);qs.append(env.state_0.joint_q.numpy().reshape(W,-1)[0,:nq].copy());b=env.state_0.body_q.numpy();x=b[env.command.body_ids_w.numpy().reshape(-1)].reshape(W,-1,7);gp=env.command.body_target_pos_w.numpy().reshape(W,-1,3);gq=env.command.body_target_quat_w.numpy().reshape(W,-1,4);poses.append(np.concatenate([x,gp,gq],axis=-1))
        if f%30==0:print(json.dumps(dict(frame=f,seconds=round(time.monotonic()-t0),height=x[0,0,2].round(3).item())),flush=True)
poses=np.asarray(poses);score=official_scores(poses);a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'poses.npz',object_pose=poses,fps=np.asarray(config.sim.fps),joint_q=np.asarray(qs),joint_ref=qref,joint_names=np.array(names),joint_q_ids=np.array(qid));result=dict(controller='zero residual',official_mp_sr=float(score['passed'].mean()),worlds=W,device=a.device,frames=F,seconds=round(time.monotonic()-t0),**{k:np.asarray(v).round(3).tolist() for k,v in score.items() if k in('mean_position_cm','mean_rotation_deg','position_cm','rotation_deg')});(a.output/'summary.json').write_text(json.dumps(result,indent=1));print(json.dumps(result)[:800],flush=True)

"""Prepare any canonical episode for the frontier loop: exported model, source tape, goal trace."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,mujoco,warp as wp
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.retargeting.evaluation_task import apply_task_protocol
from flash_chord.retargeting.physical_reference_evaluator import evaluate_native_references
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
p=argparse.ArgumentParser();p.add_argument('--parquet',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--collision-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
collision=json.loads(a.collision_from.read_text())['evaluation_task_protocol']['collision']
manifest=dict(checkpoint=str(a.checkpoint),reference=None,reference_sha256=None,evaluation_task_protocol=dict(parquet=str(a.parquet),parquet_sha256=hashlib.sha256(a.parquet.read_bytes()).hexdigest(),collision=collision))
cfg=json.loads(read_checkpoint_metadata(str(a.checkpoint))[CONFIG_METADATA_KEY]);resolved=apply_task_protocol(cfg,manifest);apply_evaluation_composition(resolved,world_count=1);resolved['env']['action']['reference_source']='native_targets'
config=instantiate_typed(OmegaConf.create(resolved['env']),RLEnvConfig);emb=instantiate_typed(OmegaConf.create(resolved['embodiment']),Embodiment);col=instantiate_typed(OmegaConf.create(resolved['collision']),CollisionPolicy)
with wp.ScopedDevice('cpu'):
    task,scene=resolved['task'],resolved['scene']
    setup=setup_scene(parquet=task['parquet'],control_fps=config.sim.fps,motion_speed=task['motion_speed'],embodiment=emb,collision=col,world_count=1,include_support=scene['support'],decompose_objects=scene['decompose_objects'],contact_friction=scene.get('contact_friction',1.),object_free_joint_damping=scene.get('object_free_joint_damping',0.),robot_reference_path=None,robot_reference_sha256=None)
    env=RLEnv(setup.scene,setup.reference,config=config);mujoco.mj_saveModel(env.solver.mj_model,str(a.output/'native_model.mjb'),None);ref=env.scene.robot_reference;Q=np.asarray(ref.joint_q,np.float32);T=np.asarray(ref.joint_target,np.float32)
print(json.dumps(dict(frames=Q.shape,targets=T.shape)),flush=True)
np.savez_compressed(a.output/'plan.npz',joint_q=Q[None],joint_target=T[None]);manifest.update(candidates=[dict(name='source_unchanged',controller='zero residual native targets')]);(a.output/'manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
result,_=evaluate_native_references(cfg,manifest,Q[None],T[None],replicas=1,device='cpu',output=a.output/'baseline');print(json.dumps(result),flush=True)

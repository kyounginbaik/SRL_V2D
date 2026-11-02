"""Replay memory the YAM recipe needs, from the real environment's dimensions."""
import sys,json
from pathlib import Path
import warp as wp,inspect
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
import flash_chord
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
pq,u=sys.argv[1:3]
with initialize_config_dir(str(Path(flash_chord.__file__).parent/'configs'),version_base='1.3'):
    c=compose('train_flash_sac',overrides=['experiment=yam_sharpa_flash_sac',f"task.parquet='{pq}'",f"embodiment.config.urdf_path._args_=['{u}']"])
c=OmegaConf.to_container(c,resolve=True);rep=c['training']['replay'];apply_evaluation_composition(c,world_count=1);cf=instantiate_typed(OmegaConf.create(c['env']),RLEnvConfig);e=instantiate_typed(OmegaConf.create(c['embodiment']),Embodiment);k=instantiate_typed(OmegaConf.create(c['collision']),CollisionPolicy)
with wp.ScopedDevice('cpu'):
    t,s=c['task'],c['scene'];st=setup_scene(parquet=t['parquet'],control_fps=cf.sim.fps,motion_speed=t['motion_speed'],embodiment=e,collision=k,world_count=1,include_support=s['support'],decompose_objects=s['decompose_objects'],robot_reference_path=None,robot_reference_sha256=None);env=RLEnv(st.scene,st.reference,config=cf)
d={n:getattr(env,n) for n in dir(env) if 'dim' in n and isinstance(getattr(type(env),n,None),property) or ('dim' in n and isinstance(getattr(env,n,None),int))}
print(json.dumps(dict(dims={k:v for k,v in d.items() if isinstance(v,int)},replay=rep)))

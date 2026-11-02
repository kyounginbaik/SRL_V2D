"""Run one upper-level round on CPU with an untrained policy: checks the wiring, not the result."""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np,jax,jax.numpy as jnp,warp as wp
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
import flash_chord
from flash_chord.configuration import instantiate_typed,resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.retargeting.yam_bilevel import YamReferenceEditor
from flash_chord.training.flash_sac.evaluation import create_actor_template
pq,u,ep,code,out=sys.argv[1:6]
with initialize_config_dir(str(Path(flash_chord.__file__).parent/'configs'),version_base='1.3'):
    c=compose('train_flash_sac',overrides=['experiment=yam_sharpa_flash_sac',f"task.parquet='{pq}'",f"embodiment.config.urdf_path._args_=['{u}']",'scene.world_count=2','training.device=cpu'])
cfg=OmegaConf.to_container(c,resolve=True);em=instantiate_typed(c.embodiment,Embodiment);col=instantiate_typed(c.collision,CollisionPolicy);tr=instantiate_typed(c.training,TrainingConfig);ec=instantiate_typed(c.env,RLEnvConfig)
with wp.ScopedDevice('cpu'):
    t,s=cfg['task'],cfg['scene'];st=setup_scene(parquet=t['parquet'],control_fps=ec.sim.fps,motion_speed=t['motion_speed'],embodiment=em,collision=col,world_count=2,include_support=s['support'],decompose_objects=s['decompose_objects'],robot_reference_path=None,robot_reference_sha256=None);env=RLEnv(st.scene,st.reference,config=ec);obs=env.observation_dim;act=env.action.action_dim
    ed=YamReferenceEditor(options=dict(engine='yam_reference',interval=1,warmup=1,candidates=3,eval_worlds=2,side='right',episode=ep,code_dir=code),setup=st,embodiment=em,collision=col,training=tr,environment_config=cfg,checkpoint_metadata={},output_dir=out)
    actor,var=create_actor_template(tr,obs,act);learner=NS(actor=NS(params=var.params,batch_stats=var.batch_stats));r=ed(NS(env=env),learner,1,None);print('replacement',type(r).__name__,flush=True)
print(open(out+'/upper_updates.jsonl').read()[:1500])

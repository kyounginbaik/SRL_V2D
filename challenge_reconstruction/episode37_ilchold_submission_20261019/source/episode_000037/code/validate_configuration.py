from pathlib import Path
import json
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment,resolve_joint_controls
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.retargeting.granularity import granularity_basis
b=Path(__file__).resolve().parents[1]
command=json.loads((b/'dry_run_seed_43.json').read_text())['training_command']
with initialize_config_dir(version_base='1.3',config_dir=str(b/'flash_chord/src/flash_chord/configs')):
 cfg=compose(config_name='train_flash_sac',overrides=command[1:])
 env=instantiate_typed(cfg.env,RLEnvConfig);robot=instantiate_typed(cfg.embodiment,Embodiment)
 controls=resolve_joint_controls(('left_wrist_x','left_wrist_y','left_wrist_z','left_wrist_rotvec','right_wrist_x','right_wrist_y','right_wrist_z','right_wrist_rotvec','left_index_MCP_FE'),robot.config.default_joint,robot.config.joint_overrides)
 assert all(c.kp==100 and c.kd==12 and c.effort_limit==90 for c in controls[:8])
 assert controls[-1].kp==1.7
 assert env.action.wrist_kp==100 and env.action.wrist_tau_max==90
 assert env.objective.hand_keypoints.weight==2
 assert env.objective.hand_joint_pos.weight==.25
 assert env.objective.hand_keypoint_palm_weight==10
 assert all(float(s.objective_weights.hand_keypoints)==2 and float(s.objective_weights.hand_joint_pos)==.25 for s in cfg.curriculum.stages)
 assert int(cfg.scene.world_count)==2048
 assert cfg.bilevel.interval==501760 and cfg.bilevel.warmup==501760
 assert cfg.training.checkpoint.save_interval_environment_steps==5033984
 assert (cfg.training.total_environment_steps-1)//cfg.bilevel.interval==100
 assert env.voc_scale==1.
 assert cfg.adaptive_voc.mode=='performance_gate'
 assert cfg.adaptive_voc.required_passes==3
 assert list(cfg.adaptive_voc.scales)==[1.,.75,.5,.25,.1,.05,.025,.01,0.]
 count=granularity_basis(335,'chunk',fps=20,chunk_seconds=1).shape[1]
 assert count==17
 result={'passed':True,'wrist_kp':100,'wrist_kd':12,'wrist_effort_limit':90,'hand_keypoint_weight':2,'finger_joint_weight':.25,'palm_relative_weight':10,'all_curriculum_stages_checked':len(cfg.curriculum.stages),'chunks':count,'lower_action_dimension':56,'upper_action_dimension':count*12,'physics_rollout_validated':False}
 (b/'configuration_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result)

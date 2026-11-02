"""Matched fresh-policy configurations for an exported physical reference."""
from copy import deepcopy
from .evaluation_task import apply_task_protocol


def fresh_reference_configuration(source,manifest,*,arm,seed,worlds,steps):
    if arm not in ('native','original') or any(not isinstance(v,int) for v in (seed,worlds,steps)) or seed<0 or worlds<2 or steps<worlds or steps%worlds:
        raise ValueError('a valid reference arm, seed, and divisible positive training budget required')
    if manifest.get('required_action_reference_source')!='native_targets':raise ValueError('verified native export required')
    config=apply_task_protocol(source,manifest);native=arm=='native'
    original_key='original_robot_reference' if 'original_robot_reference' in manifest else 'source_robot_reference'
    reference=manifest['reference'] if native else manifest[original_key]
    sha=manifest['reference_sha256'] if native else manifest[original_key+'_sha256']
    config['task'].update(robot_reference_path=reference,robot_reference_sha256=sha)
    config['scene']['world_count']=worlds
    config['training'].update(seed=seed,world_count=worlds,total_environment_steps=steps,resume=False,checkpoint_path=None,curriculum=None)
    config['training']['replay'].update(capacity=max(131072,worlds*16),minimum_size=worlds*8,batch_size=1024)
    config['training']['checkpoint'].update(save_replay=False,load_replay_path=None,load_optimizer=False,load_reward_normalizer=False,save_interval_environment_steps=steps)
    config['env']['auto_reset']=True
    config['env']['action']['reference_source']='native_targets' if native else 'geometry'
    config['env']['voc_scale']=0.
    config['env']['voc'].update(max_force=0.,max_torque=0.)
    config['env']['reset'].update(always_reset_to_first_frame=True,voc_decay_steps=0,seed=seed)
    for name in ('action','reset','voc'):
        if name in config:config[name]=deepcopy(config['env'][name])
    config['curriculum']=None;config['bilevel']=None
    config.pop('adaptation_reference_bank',None);config.pop('adaptive_voc',None)
    return config

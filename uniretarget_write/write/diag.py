"""Reset at several frames, hold the reference with zero policy action, and report what the pen, hand and board do. Inputs are the same hydra overrides as training."""
import sys, os, json, numpy as np
sys.path.insert(0, '/workspace/src')
from hydra import compose, initialize_config_dir
import warp as wp
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
clip, sup = sys.argv[1], False
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    cfg = compose('train_flash_sac', overrides=['experiment=sharpa_flash_sac', f"task.parquet=/oak/clips/{clip}/none/reference.parquet", 'logging.mode=disabled', f'scene.support={str(sup).lower()}', 'scene.world_count=4', 'objective.object_keypoints.weight=2.0', 'objective.contact_wrench_support.weight=0.0'])
env_cfg = instantiate_typed(cfg.env, RLEnvConfig); emb = instantiate_typed(cfg.embodiment, Embodiment); col = instantiate_typed(cfg.collision, CollisionPolicy); print('collision', col, 'ground', col.ground, 'sim fps', env_cfg.sim.fps, 'voc', env_cfg.voc_scale)
with wp.ScopedDevice('cuda:0'):
    s = setup_scene(parquet=cfg.task.parquet, control_fps=env_cfg.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=emb, collision=col, world_count=4, include_support=sup, decompose_objects=cfg.scene.decompose_objects, object_scale_min=cfg.scene.object_scale_min, object_scale_max=cfg.scene.object_scale_max, object_scale_seed=cfg.scene.object_scale_seed, contact_friction=cfg.scene.get('contact_friction', 1.), object_free_joint_damping=cfg.scene.get('object_free_joint_damping', 0.), source_frame_playback=bool(cfg.task.source_frame_playback), motion_start_frame=int(cfg.task.motion_start_frame), motion_end_frame=int(cfg.task.motion_end_frame))
    print('support', s.scene.support, 'bodies per world', s.scene.model.body_count // 4, 'shapes', s.scene.model.shape_count // 4)
    env = RLEnv(s.scene, s.reference, config=env_cfg); env.reset(); env.capture_step()
    n = env.world_count; per = env.model.body_count // n; ids = env.command.body_ids_w.numpy().reshape(-1); labels = list(env.model.body_label)[:per] if hasattr(env.model, 'body_label') else []; print('labels tail', labels[-3:], 'obj ids', ids[:4])
    sid = s.scene.support.body_id if s.scene.support else None
    import jax.numpy as jnp
    from flash_chord.training.environment import WarpRLEnv
    ad = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False); print('action_dim', ad.action_dim, 'ref frames', s.scene.robot_reference.num_frames)
    for f in (0, 40, 80, 100, 120, 160, 200):
        env.reset(frame_id=f); out = []
        for k in range(8):
            bq = env.state_0.body_q.numpy().reshape(n, per, 7); o = bq[:, ids[0] % per]; g = env.command.body_target_pos_w.numpy().reshape(n, -1, 3)[:, 0]
            e = float(np.linalg.norm(o[0, :3] - g[0])); line = f'k{k} objerr {e*100:6.2f}cm obj {np.round(o[0,:3],3).tolist()}'
            if sid is not None: line += f' sup {np.round(bq[0, sid], 3).tolist()}'
            out.append(line)
            r = ad.step(jnp.zeros((n, ad.action_dim), jnp.float32))
        print('frame', f, *out, sep='\n  ')

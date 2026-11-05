"""Same checkpoint, same reference, several VOC levels: does the upper level's probe agree with the VOC-off score?"""
import sys, json
src = open("/cmp/score_any.py").read().split("rows = []")[0]
exec(src)
want = [int(s) for s in sys.argv[4].split(",")]
SCALES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
rows = []
for step in want:
    path = base / f"policy_{step}.safetensors"
    state, _ = load_actor_checkpoint(str(path), template)
    meta = read_checkpoint_metadata(path); task = checkpoint_config(meta)["task"]; ref_path = task["robot_reference_path"]
    own = setup_scene(parquet=cfg.task.parquet, control_fps=env_cfg.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=embodiment,
        collision=collision, world_count=1, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
        contact_friction=cfg.scene.get("contact_friction", 1.0), object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.source_frame_playback), motion_start_frame=int(cfg.task.motion_start_frame), motion_end_frame=int(cfg.task.motion_end_frame),
        robot_reference_path=ref_path, robot_reference_sha256=task.get("robot_reference_sha256")).scene.robot_reference
    learner = SimpleNamespace(actor=SimpleNamespace(params=state.params, batch_stats=state.batch_stats))
    for s in SCALES:
        m, poses = editor.rollout(own, learner, 42, voc_probe_scale=s)
        x = poses[:, :, 0]
        raw = np.linalg.norm(x[:, :, :3] - x[:, :, 7:10], axis=2) * 100   # object against the target VOC pulls to
        row = dict(step=step, version=meta.get("robot_reference_version"), voc=s, passed=m["passed"], frontier=m["frontier"], kinds=m["kinds"],
            cm_aligned=m["cm"], deg=m["deg"], cm_raw=float(raw.mean()), start_offset_cm=float(raw[0].mean()), mid_raw_cm=float(raw[len(raw)//2].mean()), end_raw_cm=float(raw[-1].mean()))
        rows.append(row); print(json.dumps(row), flush=True)
(out / f"probe_voc_{arm}.json").write_text(json.dumps(rows, indent=1))
print("PROBE_DONE", flush=True)

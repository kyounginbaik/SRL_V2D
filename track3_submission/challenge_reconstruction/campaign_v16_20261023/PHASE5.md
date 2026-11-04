# Phase 5 (2026-10-24): head-to-head — our reference tracks vs the coworker's compose_eval tracks

User: "go on" (decide per episode which object track is closer to the truth; re-target only where the new track wins).

Tracks:
- OURS: the reference each shipped v20 component was scored against. Legacy episodes: final_recon/episode_XXXXXX/object_poses.npz
  (world_from_object, world_from_rgb camera) unless the component used a corrected source (ep1 lid-on correction:
  streamH_ep1/staging/source_corrected; ep25 board correction: streamQ_ep25/world/w3/source_corrected; ep30 corrected sweeper track:
  streamI_ep30 world). Compose episodes 14/15/16/20/25/28/30: dexagent_remaining7_20261019/input_root/challenge_reconstruction/final_recon/
  episode_XXXXXX (and the world dirs the streams built). Read each component's lineage (staged.json / provenance / input_manifest) to be sure.
- NEW: /tmp/claude-1015/-data1-SRL-V2D/da3a80c7-d8cc-4670-b1f1-f58658465433/scratchpad/compose_eval_extract/compose_eval_20261021/episode_XXXXXX/
  <obj>_trajectory.npz (T_world_obj, world = rectified-left camera at frame 0), <obj>_mesh_metric.obj (the mesh those poses refer to),
  camera_motion.npz (cam0_T_cam). README.md there lists every change they made. Videos (H.264): compose_eval_20261021_videos/h264/.
- Each track must be evaluated with ITS OWN paired mesh (poses refer to their mesh's body frame).

Evidence (independent of both trackers; same evidence for both tracks):
1. Stereo depth: per frame, render/sample the object surface in the camera, compare with the stereo depth (residual of visible
   surface points, trimmed). Existing tools: streamP_ep33/code/depthlib.py, evidence_scoop.py; streamR_ep32/code (trimmed ICP,
   stereo table fit); stereo inputs referenced by each final_recon manifest (eval_stereo_current_20261010 etc).
2. Silhouette: projected mesh vs the RGB object mask (IoU), hand pixels excluded. Find the mask source used by both pipelines
   (SAM/mask outputs referenced in the manifests); if none, say so.
3. Spoon mocap/marker balls where available (refinement_20261021_ep33/code/marker_predict33.py, ep36 marker_side.py,
   streamP_ep33 / streamR_ep32 marker scripts) — decisive for 180-deg roll questions.
4. Physical plausibility (penetration / floating) as a tie-breaker only.
Camera: both tracks must be expressed in the same camera per frame before comparing (ours: world_from_rgb; new: cam0_T_cam).
Verify the two camera paths agree (they should be the same rectified-left camera); report any mismatch.

Output per episode and object: per-window table (frames, depth residual ours/new, IoU ours/new, marker agreement), the frames
where the tracks differ by > 2 cm or > 20 deg, and a VERDICT: OURS / NEW / MIXED (with frame ranges) / UNDECIDABLE, plus a
confidence. Write h2h_<stream>/README.md and h2h_<stream>/verdicts.json
({ep: {obj: {verdict, confidence, windows:[...], notes}}}). Flag evidence that contradicts the integrator's prior expectations
(e.g. ep33 spoon roll supported by markers for ours; ep19 dust pan 40 cm gap).

Rules: read-only on everything outside your h2h dir; CPU work on partition debug (jobs <= 50 min) or interactive if light;
never touch ur-tl-* jobs; no uploads; no deletes; STATUS.md milestone lines.
Streams: h2h_spoon = eps 3, 5, 6, 15, 16, 32, 33, 35, 36, 37 (wooden_spoon + the other object in each);
h2h_other = eps 1, 8, 10, 14, 19, 20, 25, 28, 30, 38.

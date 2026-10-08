Track 3 floating Sharpa transfer, 2026-10-09

Inputs: all 20 scored eval videos (36 episode/object tracks), latest selected
reconstructed meshes, FoundationPose object tracks, and HaWoR hand tracks.
No hidden evaluation ground truth is used. Public CD-O was used in the earlier
mesh candidate selection; this does not establish hidden leaderboard quality.

Each episode is assembled in its primary object's camera/gravity frame.
Other objects use their object-to-camera poses composed with that same camera.
HaWoR hands are metric-aligned by the reconstruction pipeline, then converted
from MANO to floating Sharpa reference joint trajectories at 20 fps.

A global translation places the lowest initial object vertex at z=0. The scene
uses a ground plane, not a reconstructed support table. Object URDFs currently
use 0.3 kg and bounding-box inertia estimates. Collision-only meshes are
simplified to at most 4096 faces and decomposed with CoACD. Visual geometry and
submission meshes remain unchanged. A uniform URDF display color prevents the
Newton importer from creating a shape for every distinct vertex-color group.

Training: native FlashSAC, 50,331,648 transitions, scaling the canonical 250M assistance curriculum,
with assistance removed at 27,709,440 transitions and first-frame mixed resets
from 40,267,776 transitions. An upper contextual-bandit
FlashSAC editor proposes smooth 1-second chunk reference corrections. It edits
floating wrist translation, SO(3) wrist orientation and finger joints. Endpoints
are fixed. Translation is bounded to 3 cm per axis, rotation to 0.12 rad, and
finger offsets to 0.12 rad. Each update uses unassisted frozen-policy rollouts;
a per-object gate rejects proposals that sacrifice another object's tracking.
This is alternating optimization, not an exact bi-level hypergradient.

Runtime image: sha256:85361c8fbb2158295eb4a32abe78876852da36cb59b3c34e3bcf46460281bb8f
One Slurm-allocated GPU per episode. This cluster numbers GRES by /dev/nvidia
minor number; allocated_gpu.py translates it to a GPU UUID for Docker.
Do not select an unallocated device by observed free memory. On node200, Docker
also needs explicit --device entries for the allocated minor, nvidiactl and
nvidia-uvm; passing the allocated UUID alone did not expose CUDA correctly.

Site installation:
  Run directory: /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009
  Source checkout: /data1/track3_transfer_worktree
  Submission kit: /data1/video_to_data_challenge/v2d_submission_kit
The scripts capture this installation's paths. Adapt paths and container mounts
for a different installation. The run/code directory is the execution copy.

Order:
  load_episode.py <episode> inside v2d_task_library_loader
  retarget.sbatch <episode> using ego_recon_to_sharpa.py
  prepare_references.py, then fix_import_materials.py
  precache.sbatch (optional setup acceleration)
  train_all.sbatch (each job trains, officially evaluates, validates its episode)
  finalize.py after all 20 stage.json files are complete

run_episode.py calls the evaluator supplied by the official submission kit.
The resulting Parquets contain achieved policy rollout object poses, checkpoint
hashes and evaluator hashes. The packer enforces the public frame/object roster.
No reference trajectory is substituted for a policy rollout. A partially trained
run is not silently restarted because the current upper editor cannot resume.

Outputs:
  results/episode_XXXXXX.parquet and evaluation JSON
  submission_ready/track3_poses.csv
  submission_ready/track3_meshes.parquet
  track3_submission_ready.zip
finalize.py and monitor.py never upload to Kaggle. The user performs submission.

Validation completed before the full batch:
  20 reference object rosters and quaternion norms checked.
  Floating Sharpa exact layout: 50 scalar joints and two ball wrists.
  Numerical floating-editor bounds/endpoints and per-object gate: two tests pass.
  Existing retargeting tests: ten pass, one skipped, two FK tests deselected.
  GPU pilot: 65,536 transitions, one accepted chunk reference update.
  Official pilot export: 263 frames x two objects = 526 valid roster rows.
The pilot is a wiring test, not the final policy or a leaderboard score.

The existing GitHub repository was shared only via pushed code branches; its
visibility and collaborators were not changed. Before Kaggle upload, share it
with Nvidia-V2D-Challenge as required by the challenge submission instructions.

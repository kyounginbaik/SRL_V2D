# Floating Sharpa grasp audit, 2026-10-17

This audit distinguishes exact replay verification from task success. All pilot
and calibration runs use physical contacts, bounded joint actuators, the fixed
initial scene state, and original video timing. No object actuation, welding,
teleportation, or per-frame object pose overwrite is used.

## Findings

1. **Episode 10 had a false wrist-coordinate discontinuity.** At 4.55–4.60s,
   the right intrinsic XYZ coordinates changed from approximately
   `[-216.6, 177.6, -79.7]` to `[-38.2, -0.24, 98.8]` degrees. The actual
   rotation changes only 3.43 degrees. DexAgent's original nearest-Euler helper
   permits 2pi wraps on roll/yaw but omits pitch. Interpolating these coordinates
   commands a spurious half-turn through the cups. `code/wrist_reference.py`
   fixes the coordinate representation, preserves exact SO(3) poses, and changes
   no object, wrist-translation, finger, or timing values. Only episode 10 has
   branch changes among the five episodes. Three focused regression tests pass.

2. **Wrist servo lag is appreciable, but it is not the only grasp problem.**
   The original translation servo has kp=1000 N/m, kv=80 Ns/m, bounded at 50N.
   Rotation uses kp=50 Nm/rad, kv=5 Nms/rad, bounded at 5Nm. The velocity-related
   position error is approximately kv/kp times velocity, before inertia,
   contacts, gravity, and saturation. A controller-only sweep preserves finger
   gains, effort limits, contact geometry, masses, friction, and scene timing.
   Increasing translation kp to10000 and rotation kp to200 reduces mean left
   wrist error from1.49cm to0.34cm in ep1,1.32cm to0.16cm in ep3, and1.45cm to0.22cm
   in ep10. However all25 original-reference controller trials still fail the
   all-object MP-SR proxy. Stronger tracking can worsen impacts on a missed object.

3. **Episode 1's objects settle away from the reference before contact.** At
   0.2s, pot displacement is about1.75cm with16.7deg rotation; lid displacement
   is3.58cm with17.7deg rotation. At1.5s the differences are approximately1.9cm/
   11.3deg for the pot and3.4cm/13.9deg for the lid. The initial pot/lid collision
   has23mm penetration. An open-loop grasp aimed only at the original object
   reference is already misplaced when the hand arrives.

   The deepest initial pair is `white_pot_collision_25` against
   `white_pot_lid_collision_4` (23.07mm, predominantly lateral normal). Each
   object has32 source convex parts; this is not a single filled pot hull.
   Interior-axis probes confirm the pot cavity remains open. Source visual
   triangle meshes themselves intersect at the unchanged initial poses:
   32 horizontal planes between world z=.093m and.124m have strict line-segment
   intersections. Coarse hull chords can amplify contact by several millimetres,
   but changing decomposition alone cannot remove the source pose/shape overlap.
   No collision geometry was changed in this audit.

4. **Opposing contacts are not established at pickup.** In ep1's1–3s interval,
   the original kinematic reference has left-pot contact in31/41 frames, but the
   baseline physical hand has none. The right hand touches the lid in11/41
   physical frames and never touches it with both thumb and another finger.
   Even the kinematic lid reference first has thumb-plus-finger contacts at2.55s.
   In ep10 the physical right hand contacts the beige cup in6/41 frames and never
   with thumb plus another finger, versus40/41 reference contact frames.
   These are geometric contact observations, not a claim of force closure.

5. **The kinematic references themselves penetrate object colliders.** At
   several intended grip frames, fingers penetrate pot/lid/cup colliders by
   roughly1.1–1.4cm. Finger actuator tracking is otherwise fairly accurate:
   mean joint-angle discrepancies are typically1.2–3.6deg. Increasing all finger
   force limits is therefore not a demonstrated solution. Grasp position,
   thumb opposition, closure timing, and actual object state need correction.

## Consequences for refinement

- Apply the exact equivalent-angle correction before proposing or evaluating
  wrist trajectories. `continuous_references/` contains all five exports.
- Use actual predicted object poses during approach and pickup, including
  passive settling, while retaining the original object path as the task target.
- Search wrist position/orientation and finger closure together. Evaluate an
  early lift and contact persistence before optimizing later placement motions.
- Keep effort caps fixed while tuning gains. A moderate kp5000/kv80 translation,
  kp200/kv5 rotation controller is a useful candidate, not a proven universally
  successful controller. With the angle fix it makes ep10's blue cup pass the
  proxy thresholds (1.58cm,4.89deg), but the beige cup still misses pickup.
- Complete full-horizon physical replay and all-object MP-SR evaluation for any
  selected trajectory. Neither controller tracking accuracy nor one successful
  object is sufficient for episode success.

## Evidence

- `controller_sweep/summary.json`:25 unchanged-reference controller trials.
- `controller_sweep_continuous/summary.json`: equivalent-angle reference trials.
- `continuous_references/report.json`: per-frame pose-preservation audit.
- `visual_audit/episode_000001_timeline.jpg` and `episode_000010_timeline.jpg`:
  original video, original reference, baseline physics, refined physics.
- `visual_audit/episode_000001_contacts.json` and `episode_000010_contacts.json`:
  every-frame contact identities and penetrations. The optional closest-surface
  distances are secondary diagnostics; zero from a mesh-distance query alone
  is not counted as contact.
- `code/tests/test_wrist_reference.py`: pitch-crossing and all-axis wrap tests.
- `visual_audit/episode_000001_initial_collision.json`: exact offending geom
  pairs, source surface distances, hull bounds and cavity occupancy probes.
- `visual_audit/episode_000001_initial_sections.png` and
  `episode_000001_visual_intersections.json`: source visual/collision
  cross-sections and strict visual-mesh intersection evidence.

The calibration directories are isolated from `pilot_v2` and `final_recon`.
The original source assets and pilot records remain unchanged.

## Episode 1 follow-up isolation

The following experiments preserve the original object targets and initial
states, mass, contact friction, and torque/force limits. Every reported trial
was run for the complete original timeline and independently replay-verified.
None has passed all-object reference-proxy MP-SR.

- `settled_seed/`: four tests corrected wrist approach for measured passive
  object settling, using either original or geometric finger references and
  either original or stronger wrist gains. Settling compensation alone did
  not establish a grasp.
- `lid_seed/`: handle-specific thumb/index IK at the predicted settled lid
  pose, with whole-hand collision residuals, produced 9 geometrically screened
  poses out of 12 attempts. The best three physical tests still missed pickup.
  Their command penetration was about 2.3 mm, compared with 13–16 mm in rejected
  earlier geometric grasps. Geometric feasibility does not establish force closure.
- `lid_pressure/`: twelve bounded generic thumb/index closure offsets failed
  to retain the lid; full-episode lid position error remained 12.39–13.29 cm.
- `wrist_feedforward/`: nine gravity and/or reference-velocity feedforward
  variants altered only wrist servo targets, with the original actuator caps.
  No external forces were applied. They did not establish pickup; lid error
  remained 12.62–13.02 cm.
- `finger_gain/`: six tests used 2x or 4x finger stiffness and sqrt(2)x or 2x
  servo damping on identical references. Joint friction and torque caps were
  unchanged. At 4x stiffness, right-hand mean finger tracking error decreased
  to 1.10–1.16 degrees, but lid error remained 12.32–12.76 cm across the sweep.
  Thumb-plus-index contact did not persist. Raising gains alone is therefore
  insufficient for these grasp proposals.

The force audit at 1.7 s found no finger actuator at its torque cap. Several
distal servos produced forces near their joint-friction deadband. Increasing
stiffness improves those joint errors, but changing the direction and geometry
of pad closure is still necessary. Object-normal pad preload is tested
separately in `lid_normal/`; its success must be assessed from physical replay,
not from commanded penetration or fingertip proximity.

All six `lid_normal/` tests also failed (lid position error 12.63–12.94 cm).
Thumb IK achieved the requested 3/6 mm inward travel, while index achieved only
−0.12 to 3.24 mm with a fixed wrist. Nine `lid_dwell/` tests then advanced the
robot's arrival to 1.5 s, held its grasp command until 1.7/1.9/2.1 s, and joined
the original robot path over 0.4 s. The simulation clock and object target were
unchanged. These also failed pickup (lid error 12.76–13.92 cm).

The actual lid matches its passive forecast until 1.4 s in handle candidate 1.
First finger contact then changes the object pose: by 1.6 s, translation differs
by approximately 4.7 mm and orientation by 6.95 degrees; by 1.7 s, rotation
differs by 9.55 degrees. This moves the small handle away from the other finger
before capture. A static pose derived from passive settling cannot anticipate
this interaction. The next controller test therefore uses physical object-pose
feedback and staged opposing-finger/thumb closure; it must still pass independent
control replay and the unchanged full-timeline object metrics.

## Feedback acquisition and actual contact geometry

`lid_acquisition_v2/` uses the shared contact-acquisition controller: observe
the actual unactuated lid SE(3), approach with the thumb held open, close it
after index contact, then briefly dwell and transport. Controls are recorded
at 500 Hz and independently replayed. Physical initialization is asserted
bitwise equal to the frozen scene keyframe. Desired-reference/keyframe XML
rounding is logged separately; it does not change the physical initial state.

Candidate 2 with 10 mm thumb opening produced a two-finger force event at
1.800 s and started transport at 1.920 s. The lid rose 25.9 mm and was more
than 1 cm above its pregrasp height without table contact for 0.8 s. However,
the actual thumb/index contact normals had dot products +0.26 to +0.47: both
pressed from the same general side. Thus this was not an antipodal grasp.
Contact was lost by 2.4 s, and full-episode lid position error was 12.81 cm.
The rendered video and contact/retention audits are stored with that candidate.

`lid_retention/` preserves the captured wrist/object transform and closed
finger posture through transport. The contact-only gate still fails
(12.41 cm lid error); a gate checking actual normal opposition and the
contact-to-contact line inside both Coulomb friction cones correctly refuses
the earlier apparent capture. Neither is a success.

`lid_collision_grasps/` samples exposed surfaces of the exact 32 compiled
collision pieces, excluding faces buried in another convex part. Strongly
opposing planned pairs then undergo whole-hand IK and actual MuJoCo contact
screening. Nine of sixteen solved poses have acceptable collision clearance
and opposing actual thumb/index normals. Eight feedback executions of the
four selected poses (`lid_collision_feedback/`) nevertheless fail acquisition.
At 1.5–2.2 s, the commanded index touches the actual lid while the measured
index generally does not. Wrist gravity sag is approximately 2.5 mm, with
additional finger-servo error. A bounded follow-up combines right-wrist
gravity feedforward and right-finger stiffness calibration on these new poses;
it is separate from the earlier tests using non-opposing visual-mesh grasps.

All six combined calibration tests in `lid_feedback_calibration/` completed
with valid replay and still failed qualified acquisition (lid mean position
error 12.44–13.36 cm). The 4x finger-gain variant changes only the right hand;
gravity feedforward is applied as a bounded right-wrist servo-target offset,
never as external force. The force/torque caps, friction, meshes, masses,
original object targets, and physical initial state remain unchanged.

The evidence supports contact-seeking acquisition rather than more gain
sweeps: the controller must bring the measured opposing finger to the real
collision patch, then close the thumb without pushing the object away. Neither
static IK, commanded penetration, a two-finger force event, nor transient
airborne motion is sufficient to certify a stable grasp or submission success.

## Bounded contact-seeking test

`lid_contact_seeking/episode_000001/candidate_00_open10/` adds a measured
index-to-lid closest-point approach after the nominal approach finishes.
The additional wrist path is bounded at 0.02 m/s and 1 cm total; force-bearing
index contact stops seeking. The exact original state and simulation prefix
through 1.45 s are bitwise identical to the corresponding baseline (qpos,
qvel, and controls). The full 500 Hz recording passes independent replay.

Seeking stops at 1.492 s after only 0.84 mm of extra travel, with 4.52 N of
measured index contact force (13.13 N peak later). No hand/table force is
observed. This confirms that a very small measured correction can reach the
physical collision patch. It does not establish a grasp: the transient contact
does not satisfy the inherited 20 ms contact-confirmation gate for thumb
closure, so no qualified capture occurs. Lid error is 12.72 cm / 36.89 degrees;
pot error is 7.17 cm / 15.32 degrees; full-object MP-SR remains zero. No object
actuation, state edits, gain/cap changes, or episode-clock changes are used.

`controlled_test_summary.json` records the exact displacement bounds,
force observations, prefix identity checks, and complete metrics. This case
must not be promoted on the basis of the contact force event.

The isolated follow-up in `lid_contact_seeking_atomic/` fixes the transition:
seeking stops and thumb closure starts on the same force-bearing event at
1.492 s. The index offset is then held in the object's frame. Contact is lost
at 1.500 s, only 8 ms later, before the nominal 150 ms thumb closure completes
at 1.642 s. Index contact is present for 18.7% of that closing interval, and
there is no qualified capture or meaningful lift. Final lid error is
12.74 cm / 40.25 degrees; pot error is 7.20 cm / 18.45 degrees; MP-SR is zero.

This follow-up also passes independent replay, preserves the exact physical
initial state and bitwise baseline prefix through 1.45 s, respects 0.02 m/s
extra-motion speed, uses only 0.84 mm of the 1 cm travel budget, and records
zero right-hand/table force. It is a negative result: reaching a collision
patch momentarily is insufficient to retain contact while the thumb closes.
The earlier controller source is archived with the first test. A final,
two-case comparison tests whether the wide thumb opening causes this loss.

## Final thumb-preposition comparison

CPU array 47775 executes the same collision-qualified grasp and bounded index
seeking, with the thumb either already closed during the existing smooth
approach or opened by 2 mm (1.82 mm achieved by finger IK). It changes no gains,
effort caps, object initial states, object reference poses, or episode timing.
The exact physical initial state and baseline prefix through acquisition start
at 1.1 s are bitwise identical in qpos, qvel, and controls. Both complete
500 Hz recordings pass independent physics replay.

| Thumb opening | Extra seek | First index contact | First contact loss | Longest actual opposing contact | Lid mean position / orientation error | Full MP-SR |
| --- | --- | --- | --- | --- | --- | --- |
| 0 mm | 0.76 mm | 1.488 s | 1.496 s | 42 ms | 28.62 cm / 46.26 deg | 0 |
| 2 mm requested | 1.20 mm | 1.510 s | 1.516 s | 18 ms | 12.37 cm / 41.67 deg | 0 |

Actual thumb/index contacts do briefly satisfy normal opposition and both
Coulomb friction-cone tests. They never sustain the required 60 ms capture
interval, so neither controller enters capture or transport. The closed-thumb
case subsequently loses the lid from the table; its final height is -5.28 m
at 16.7 s. That is a physical failure, not retained lifting. The 2 mm case
also fails despite a small temporary height increase. Both record zero
right-hand/table normal force, and extra seeking remains at or below 0.02 m/s.

Thus eliminating delayed thumb travel improves momentary opposition but does
not solve retention. The remaining failure is not explained solely by a large
thumb opening, a wrong contact normal, or table blockage of this right-hand
pose. Contact evolves and breaks under the live object-following trajectory.
These tests provide no successful episode-1 lid grasp and must not replace the
selected submission candidate. No further trials are launched in this sweep.

Evidence: `lid_thumb_preposition/comparison_summary.json`, each candidate's
`controlled_test_summary.json`, `execution/actual_contact_geometry.json`, and
the independent replay report referenced by `report.json`.

# Stream P8 (v23, PHASE8.md): episode 35 (single wooden_spoon), pinch / hold re-plan, 2026-10-28

Goal (coordinator brief): spoon <= 1.2 cm / 12 deg mean vs `target_final_v5`, build-v2 exact. Starting point K8_ep35_ilc2 1.27 / 13.5.
Goal later relaxed by the user to <= 1.3 cm / 15 deg; K8_ep35_ilc2 meets it -> stream stopped on coordinator request.

## Result
**Nothing here beats K8_ep35_ilc2 (1.27 / 13.5, build-v2 exact, staging_K8_35b) -> it stays the ep35 component.** No P8 tag was staged
or verified (no candidate was better on both axes). Best P8 actors (official evaluate_policy_component = search value, not build-v2):
| actor (runs/...) | spoon cm / deg | early 0-9.5 s deg | late 9.5-21 s deg | note |
|---|---|---|---|---|
| (K8_ep35_ilc2 = K8 runs/i35_b) | 1.27 / 13.52 | 20.6 | 7.7 | reference |
| iA2b/actor | 1.49 / 12.53 | 16.8 | 9.0 | lower deg, but pos 1.49 > 1.3 |
| iA2d/actor | 1.47 / 12.68 | | | |
| iA2e/actor | 1.47 / 12.80 | | | |
| i_Vb (search, cancelled) | 1.49 / 13.81 | | | |

## Findings (numbers)
1. **ep35 already uses a fingertip pinch** (code/contacts.py on the K8 rollout): thumb-DP + index-DP 1.5-5.3 s (same pickup planner
   lineage as ep37: native_mpc -> pickup_search -> ...). At 4.5-5.3 s the LEFT hand touches the spoon and it reseats into thumb-PP /
   palm (6-18 s). So "use a pinch like ep37" does not change the grasp class; ep37's pinch was benign because of the TARGET MOTION:
   ep37 hold 3 s, max rotation from lift 23 deg, 4 frames > 200 deg/s; ep35 hold 17.5 s, rotation up to 179 deg from lift,
   32 frames > 200 deg/s (p99 448 deg/s), long-axis rolls of ~-110 deg and back within 1.5 s at 6.8-8.4 s.
2. **Where the chaos comes from** (code/inhand.py, contactsQ.py): early edits leave the in-hand pose almost unchanged up to 10 s
   (<= 14 deg), then the LEFT hand pressing the spoon at 9.2-11.8 s (left thumb MC / palm) flips it to a different in-hand seat
   (40-55 deg in-hand rotation, kept to 18.5 s). K8's late hold (7.7 deg) is one lucky outcome of that bifurcation:
   92-run grid of +-1-3 cm left-wrist offsets (runs/left3) and squeeze grids (runs/sq1: 0.05-0.35 rad, thumb/fingers) all lose it
   (late 9-100 deg); roll pulses about the spoon axis (runs/roll1) improve 6-9.5 s (24.9 -> 20.2 deg) but lose the late hold.
3. **Moving the left wrist 4 cm away (x,y) from 8.8 s** removes the bifurcation: late hold becomes robust to early edits
   (runs/left2; bakeA1 1.90/17.8, bakeA2 = + roll -30 deg 7-8.2 s 1.96/19.0). Windowed ilcK on bakeA2 then reaches early 16.8 deg
   (best seen in any ep35 rollout), but the left-free hold is worse late: best late 9.0 deg (vs 7.7) and pos rises (8 s window 4.1 cm)
   -> 1.49 / 12.53. Mid windows 7.4-14.4 s stay non-improvable by ILC (0 better candidates); 2.8-5.6 s ILC drops the spoon.
   Squeeze on the left-free actor (runs/sq2, 36 runs) only worsens it. Three more early variants + ILC chains (i_Va/Vb/Vc): 1.49-1.70.
4. Floor arithmetic: with K8's late 7.7 deg, 12 deg needs early <= 17.2 deg; with the left-free late (>= 9.0) it needs early <= 15.6.
   Best early with any kept hold: 16.8 (iA2b). Neither combination exists physically in ~600 rollouts here.

## Code (code/; K8 ilcK.py / tlibK.py used unchanged by import)
contacts.py, contactsQ.py (hand-spoon contacts), roll_diag.py (body-frame error decomposition), gridP.py (+grid.sbatch: parallel
phase-table edit grids incl. roll about the executed spoon axis (chart-aware) and LEFT-wrist offsets), inhand.py (in-hand pose vs base),
bake.py (bake edits into a checkpoint + official eval via ilcK.save_actor), res.sbatch / chain.sbatch (ilcK runs). Searches on node200.
Specs in specs/, grids/ILC runs in runs/. Nothing staged, nothing uploaded, no commits.

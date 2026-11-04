# Episode 25 re-plan (2026-10-22) - our own board meshes only (no scans)

Status: NO staged component. A passing open-loop teacher exists but no learned actor passes; board 2 success is knife-edge.

1. Slot collision from our own evidence (code/build_slot_collision.py, code/cut_slot_parts.py): our visual meshes and the
   slots_v3 collision had no assembly slot (board 2 could not descend into board 1). The assembled cross in our reconstruction
   (frames >= 17 s) gives slot position/opening side; the slot (width = other board's collision thickness + 3 mm, half height
   deep, board 1 opens up, board 2 opens down) was clipped out of the EXISTING validated collision parts
   (collision/wooden_piece_{1,2}_cut3mm). Original inertia kept (KEEP_INERTIA) - recomputing it flipped board 1's topple.
2. Scenes: scenes/pf4_cut3mm_keepI.xml (best old teacher's scene with slot collision). Old teacher replay in it: 3.47/32.1, 10.09/50.4.
3. Chained retention re-plans (search/): best passing teachers (proxy MP-SR 1, original reference):
   c4_L_14.0_17.8  wooden_piece_1 2.95 cm / 29.4 deg, wooden_piece_2 2.96 / 23.8
   c5_R_12.0_17.5  2.97 / 28.9, 2.95 / 23.1
4. Actor grid on c4_L_14.0_17.8 (12 configs): all fail; board 2 lands 9-13 cm off (best: c1024 p0.003: 3.22/31.6, 8.93/56.6).
5. Robust (worst-of-3 finger-perturbed) re-plans of the board-2 window: board 2 always 7.4-8.1 cm -> the passing board-2
   outcome is a knife-edge; in every plan board 2 is never grasped (right hand flicks it at ~9.5 s; it lands on board 1).
Blocker: a real right-hand grasp of the flat thin board 2 + controlled insertion is needed (a new pickup design), which the
current plan family cannot provide.

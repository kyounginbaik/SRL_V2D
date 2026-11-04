## v21 (2026-10-24)

v20 with references checked frame by frame against independent evidence (mocap marker balls on the objects, RGB silhouettes, stereo depth) and corrected only where that evidence supports it. All meshes are our own reconstructions. All rows carry the build v2 evaluator hash pair and commit URL. Local errors are against the corrected references (cm / deg).

- Episode 15: spoon re-targeted to the refined spoon track (2-4 cm along the handle; marker balls 10 px vs 77 px); the robot now releases the spoon in the rack. Spoon 0.80 / 5.8, rack 0.32 / 2.3.
- Episode 16: spoon re-targeted the same way (balls 9 px vs 44 px). Spoon 1.25 / 12.7, rack 0.26 / 1.9.
- Episode 32: the spoon roll is corrected in three windows where the marker balls show a near half-turn; the robot follows it with a power grasp. Cup 1.21 / 15.1, spoon 1.60 / 13.7.
- Episode 33: the spoon roll is corrected at 6.7-8.25 s (marker balls 3-12 px vs 117-120 px); the robot follows it. Cup 0.75 / 5.0, spoon 2.11 / 19.3.
- Episode 25: board 2's position corrected at 2.9-8.5 s where stereo supports it; robot actions unchanged. Boards 2.02 / 13.1 and 2.30 / 11.6.
- All other episodes identical to v20.

Videos for episodes 15, 16, 32 and 33 compare the v17 rollout with the v21 rollout. Not uploaded.

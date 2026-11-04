DexAgent floating Sharpa: pilot_v2, seed 44
Reference-proxy MP-SR, not a leaderboard score. Complete execution and independent control replay are checked separately; passing these checks does not certify penetration-free motion.
Completed 5/5; valid paired metrics 5.

| Episode | Status | MP-SR baseline → refined | Execution/replay baseline/refined | Selected cost change | Search / total min |
|---|---|---|---|---|---|
| 000001 | complete | 0.00 → 0.00 | pass/pass | 2.70 → 2.42 | 2.78 / 3.79 |
| 000003 | complete | 0.00 → 0.00 | pass/pass | 2.42 → 2.22 | 1.65 / 2.31 |
| 000005 | complete | 0.00 → 0.00 | pass/pass | 3.24 → 3.10 | 1.57 / 2.25 |
| 000008 | complete | 0.00 → 0.00 | pass/pass | 3.25 → 1.50 | 1.49 / 2.13 |
| 000010 | complete | 0.00 → 0.00 | pass/pass | 136.13 → 65.35 | 0.98 / 1.54 |

Per-object errors below use official-style shared initial-scene alignment; multiple worlds are separated by `/`.

| Episode / object | Mean position cm: baseline → refined | Mean orientation deg: baseline → refined |
|---|---|---|
| 000001 / white_pot | 6.74 → 6.82 | 12.15 → 13.95 |
| 000001 / white_pot_lid | 12.79 → 12.45 | 33.09 → 34.28 |
| 000003 / white_pot | 4.57 → 4.52 | 28.31 → 25.15 |
| 000003 / wooden_spoon | 13.40 → 12.90 | 38.99 → 26.20 |
| 000005 / white_pot | 8.47 → 8.53 | 33.13 → 28.17 |
| 000005 / wooden_spoon | 16.05 → 15.98 | 42.98 → 44.36 |
| 000008 / beige_cup | 10.19 → 18.39 | 32.20 → 43.87 |
| 000008 / white_pot | 3.74 → 3.22 | 12.22 → 19.29 |
| 000010 / beige_cup | 641.18 → 556.84 | 41.28 → 38.70 |
| 000010 / blue_cup | 657.36 → 540.08 | 39.16 → 36.94 |

episode_000001: selected commands changed 177/335 frames; left wrist mean/max 0.68/4.15 cm, finger mean/max 0.41/6.20 deg; right wrist mean/max 0.89/5.04 cm, finger mean/max 0.45/7.97 deg.
episode_000003: selected commands changed 154/313 frames; left wrist mean/max 1.01/4.69 cm, finger mean/max 0.48/9.15 deg; right wrist mean/max 0.90/4.53 cm, finger mean/max 0.54/10.23 deg.
episode_000005: selected commands changed 218/288 frames; left wrist mean/max 1.25/4.07 cm, finger mean/max 0.80/14.38 deg; right wrist mean/max 1.18/4.06 cm, finger mean/max 0.68/12.22 deg.
episode_000008: selected commands changed 195/219 frames; left wrist mean/max 1.78/5.18 cm, finger mean/max 1.13/9.59 deg; right wrist mean/max 1.60/4.15 cm, finger mean/max 1.13/11.39 deg.
episode_000010: selected commands changed 109/161 frames; left wrist mean/max 1.18/4.70 cm, finger mean/max 0.77/6.07 deg; right wrist mean/max 1.14/3.98 cm, finger mean/max 0.88/8.26 deg.

All-requested MP-SR: 0.00 → 0.00; physics-gated: 0.00 → 0.00.

Interpretation: the search objective includes contact-force and penetration penalties; reducing it does not necessarily improve MP-SR. These results cover the floating-Sharpa MPC module port with reference poses as proxy ground truth.
Scene limitations: the finite tabletop has no room floor beneath it, so a dropped object's continued fall can inflate distance errors. Episode 1 also retains the supplied pot/lid mesh overlap. See the [experiment documentation](../README.md) for source assumptions, drop audits and comparison videos.

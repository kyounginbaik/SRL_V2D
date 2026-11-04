# Variant A (2026-10-30): natural-motion smoother on all eval episodes except 32

Deliverables (base `selected20_final_v8_g10b`, same format; ep32 and every non-pose file byte-identical):

| bundle | zip (sha256) | rule |
|---|---|---|
| `A_smooth_mild/` | `A_smooth_mild.zip` (dbbdd6feed1a154d6382c6e3318941af40163fb167592399c83a36c41c892a3f) | robust Whittaker lam 8 + tight image guard (IoU 0.05, depth 3 mm, ball 3 px) |
| `A_smooth_strong/` | `A_smooth_strong.zip` (d5ca9e3fff8f989b43e2860de9c5d6477b83e1aea4ca7411ce2a844b74be68c0) | robust Whittaker lam 32 + loose image guard (IoU 0.1, depth 6 mm, ball 6 px) |

Each bundle has `README.md` (rule, public table, per-object eval QC) and `manifest.json` (changed objects, kept jumps,
steps before/after, sha256 of every file and of each changed npz before/after, parameters, public and eval QC). Changed:
`object_poses.npz` of episodes 1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 33, 35, 36, 37, 38 (all 34 objects). Frame 0 is unchanged everywhere.

## Sweep (same rule on public 10 + held-out 10; unchanged scorer, GT only inside it)

Eval QC: team `qc.py` unchanged + spoon ball reprojection, on frames the variant changed (GT-free).

| tag | rule | MP-SR | mean pos cm | mean ori deg | MPPE | AUC | obj worse > .05 cm | eval IoU | eval abs depth mm | eval frames IoU drop > .1 | eval steps > 15 deg | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8_base | base | 20/20 | 1.114 | 5.23 | 1.171 | 0.9312 | - | - | - | - | 147 | baseline |
| w2 | unguarded lam 2 | 20/20 | 1.093 | 5.14 | 1.148 | 0.9325 | 0 | 0.750 -> 0.745 | 8.95 -> 9.30 | 142 | 44 | rejected: eval QC (ep5 spoon IoU 0.76 -> 0.70, ep6 spoon, ep19 sweeper) |
| w8 | unguarded lam 8 | 20/20 | 1.090 | 5.11 | 1.145 | 0.9327 | 0 | 0.751 -> 0.738 | 8.65 -> 9.72 | 301 | 25 | rejected: eval QC |
| w32 | unguarded lam 32 | 20/20 | 1.100 | 5.11 | 1.155 | 0.9314 | 3 | 0.752 -> 0.726 | 8.41 -> 10.63 | 615 | 9 | rejected |
| **g8** | lam 8, tight guard | 20/20 | 1.096 | 5.14 | 1.151 | 0.9322 | 0 | 0.751 -> 0.753 | 8.38 -> 8.25 | 0 | 71 | **mild** |
| g32 | lam 32, tight guard | 20/20 | 1.094 | 5.15 | 1.150 | 0.9320 | 0 | (guard: no worse) | | | 69 | passes; same smoothness as g8 |
| **s32** | lam 32, loose guard | 20/20 | 1.095 | 5.12 | 1.152 | 0.9320 | 0 | 0.753 -> 0.752 | 8.30 -> 8.25 | 0 | 45 | **strong** |
| s128 | lam 128, loose guard | 20/20 | 1.102 | 5.15 | 1.161 | 0.9309 | 3 | 0.754 -> 0.752 | 8.25 -> 8.45 | 0 | 42 | not chosen (public worse than s32, AUC < base) |

SP-SR 20/20 for all. Max object delta vs base: g8 +0.02 cm / +0.09 deg, s32 +0.04 cm / +0.15 deg.
Unguarded smoothing improves public slightly but moves image-consistent fast motion (ep5 stirring spoon at 2-5 cm/frame,
IoU 0.85-0.95, ball 2-6 px) off the image, so the guard is required. Eval large steps (v8_g10b -> mild / strong):
rotation steps > 30 deg 10 -> 3 / 3 (the 3 left are kept persistent jumps: ep19 sweeper in-fist window), > 15 deg
147 -> 71 / 45, centre steps > 3 cm 92 -> 41 / 30, angular-acceleration events > 20 deg/frame^2 62 -> 11 / 7.

## How it was run
- Code `A_work/code/`: `slib.py` (stats + smoother), `guard_clip.py` (image guard), `gen_all.sh`, `score_tags.sh` / `score.sh`
  (unchanged `tools/reconstruction_audit/score_bundle.py` in a mirrored episode root `A_work/pub/`), `qc_tag.sh` (unchanged `qc.py`),
  `ballqc.py`, `compare.py`, `qc_compare.py`, `build_bundle.py`, `make_readme.py`.
- Python: our 3.11 venv + teammate venv site-packages via `code/genv.sh` (re-scoring r8_base ep11 reproduced the stored score exactly).
- Scores: `A_work/compare_<tag>.json`, `A_work/compare_g8_s32.txt`; eval QC: `A_work/qc_eval_<tag>.json`, per episode `A_work/eval/<tag>/episode_*/{qc.json,ballqc.json,guard_report.json}`.
- Nothing in `/data1/SRL_V2D` was written; no upload, no commit.

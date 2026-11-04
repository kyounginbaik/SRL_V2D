# B14_family_consensus (Track 3 recon variant B2 on v14_g17, 2026-10-30)

Base: `selected20_final_v14_g17`, same format. Byte-identical copy of v14_g17; only `object_poses.npz` of the changed episodes is rewritten (same keys and dtypes, compressed; untouched objects and all other keys are identical to v14). Meshes unchanged; ep32 is v14_g17. No GT, no eval GT, no scanned meshes.

## Rule (identical to the v8-based variant B; code `B14_work/code/v14lib.py` + `B_work/code/variants.py`)
Shipped variant-B rule, unchanged: per frame, soft trust-region weighted SE(3) mean of the base and the type's good families (B_work/rules.json). Weight exp(-d^2/2), d = max(dp/1.5cm, dr/15deg) vs the base, base weight 1; half-turn twins for sweeper/pieces; cups/lid position only. Spoon, blue cup, lid, piece_2: base. The same weights are applied to world_from_object and camera_from_object.

## Clock
Every family track is first mapped onto the v14 clock with the exact G15 operator the team used for the base: world = retime(G15 camera @ family camera_from_object, delta_ep), camera = retime(family camera_from_object, delta_ep) (streamG15/code/retime.py shift, out/sync_delta_t.json; ep20 piecewise 0.925/1.85). Checked: the same operator on v8_g10b reproduces v14_g17 exactly for every slot a rule touches (max abs diff 0). Missing family frames: B1 filled with the pre-G15 base, B2 weight 0 (mask carried through the shift).

GT-free check: without the retime, family tracks lag the v14 base by a median 0.875 frames on eval (0.875 public); after it, 0.0 (0.0), and the moving-frame gap to the base drops 1.00 -> 0.51 cm on eval (0.76 -> 0.30 public).

## Public / held-out score of exactly this rule (unchanged scorer; reference g15_camrt)

| set | MP-SR | SP-SR | pos cm | ori deg | MPPE | AUC |
|---|---|---|---|---|---|---|
| v14 reference all | 20/20 | 20/20 | 0.999 | 4.95 | 1.0641 | 0.9437 |
| **B2 all** | 20/20 | 20/20 | 0.989 | 4.70 | 1.0415 | 0.9452 |
| B2 without family retime all | 20/20 | 20/20 | 0.985 | 4.77 | 1.0409 | 0.9441 |
| v14 reference public | 10/10 | 10/10 | 1.072 | 3.79 | 1.0931 | 0.9496 |
| **B2 public** | 10/10 | 10/10 | 1.061 | 3.47 | 1.0607 | 0.9517 |
| B2 without family retime public | 10/10 | 10/10 | 1.066 | 3.50 | 1.0631 | 0.9504 |
| v14 reference heldout | 10/10 | 10/10 | 0.921 | 6.18 | 1.0351 | 0.9379 |
| **B2 heldout** | 10/10 | 10/10 | 0.912 | 6.00 | 1.0223 | 0.9386 |
| B2 without family retime heldout | 10/10 | 10/10 | 0.899 | 6.12 | 1.0188 | 0.9379 |

2-fold check: public (rule from held-out only): MPPE 1.0931 -> 1.0616, AUC 0.9496 -> 0.9525, MP 10/10; heldout (rule from public only): MPPE 1.0351 -> 1.0310, AUC 0.9379 -> 0.9382, MP 10/10

## Changed eval episodes: [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 28, 30, 38]

| ep | object | source | mean/max move vs v14 cm | max deg | IoU median v14->new | paired dIoU | mean rot step deg v14->new | max rot step |
|---|---|---|---|---|---|---|---|---|
| 1 | white_pot | consensus(base+raw) | 0.07/0.59 | 4.6 | 0.805->0.818 | +0.0057 | 1.41->1.31 | 7.7->6.6 |
| 3 | white_pot | consensus(base+raw) | 0.11/0.46 | 3.6 | 0.773->0.776 | +0.0024 | 1.00->0.94 | 7.5->4.8 |
| 5 | white_pot | consensus(base+raw) | 0.17/0.59 | 5.9 | 0.720->0.723 | +0.0052 | 2.38->2.36 | 15.2->12.3 |
| 6 | white_pot | consensus(base+raw) | 0.05/0.59 | 5.1 | 0.746->0.746 | +0.0018 | 1.97->1.90 | 27.1->24.0 |
| 8 | beige_cup | consensus(base+raw) | 0.15/0.59 | 0.0 | 0.661->0.662 | +0.0045 | 6.18->6.18 | 49.7->49.7 |
| 8 | white_pot | consensus(base+raw) | 0.10/0.38 | 3.2 | 0.761->0.762 | +0.0013 | 0.91->0.88 | 6.0->6.0 |
| 10 | beige_cup | consensus(base+raw) | 0.18/0.52 | 0.0 | 0.599->0.599 | -0.0030 | 2.81->2.81 | 42.9->42.9 |
| 14 | plastic_dish_rack | consensus(base+multimap+pointmap_cotracker+raw) | 0.16/0.49 | 3.1 | 0.877->0.877 | -0.0010 | 0.55->0.29 | 3.9->1.7 |
| 15 | plastic_dish_rack | consensus(base+multimap+pointmap_cotracker+raw) | 0.09/0.17 | 1.3 | 0.886->0.887 | +0.0016 | 0.35->0.22 | 1.6->0.8 |
| 16 | plastic_dish_rack | consensus(base+multimap+pointmap_cotracker+raw) | 0.11/0.53 | 2.1 | 0.885->0.883 | +0.0007 | 0.48->0.29 | 4.4->2.0 |
| 19 | mini_sweeper | consensus(base+multimap) | 0.08/0.59 | 5.9 | 0.723->0.726 | +0.0019 | 4.71->4.61 | 140.4->144.3 |
| 19 | mini_dust_pan | consensus(base+multimap+raw) | 0.12/0.58 | 4.5 | 0.771->0.775 | +0.0049 | 2.05->2.05 | 14.3->15.4 |
| 20 | water_pitcher | consensus(base+raw) | 0.05/0.59 | 5.8 | 0.494->0.492 | -0.0010 | 2.05->2.00 | 18.5->14.6 |
| 28 | plastic_dish_rack | consensus(base+multimap+pointmap_cotracker+raw) | 0.16/0.48 | 1.8 | 0.875->0.876 | +0.0021 | 0.42->0.27 | 2.6->1.0 |
| 28 | white_pot | consensus(base+raw) | 0.02/0.34 | 5.6 | 0.821->0.819 | -0.0002 | 2.41->2.37 | 13.8->13.8 |
| 30 | mini_sweeper | consensus(base+multimap) | 0.13/0.59 | 5.6 | 0.793->0.788 | -0.0038 | 2.22->1.88 | 15.7->9.7 |
| 30 | plastic_dish_rack | consensus(base+multimap+pointmap_cotracker+raw) | 0.15/0.34 | 3.2 | 0.860->0.863 | +0.0035 | 0.56->0.29 | 3.4->1.1 |
| 38 | mini_dust_pan | consensus(base+multimap+raw) | 0.15/0.35 | 3.5 | 0.794->0.794 | -0.0001 | 1.95->2.01 | 16.7->15.0 |

Not retargeted: the v29 rollouts target v14_g17; these bundles only matter if components are re-retargeted onto them.
Details: `manifest.json` (sources, sha256 of every file, scores, QC, lag check). Work folder: `/home/hansol/v2d/outputs/variants_20261030/B14_work`.

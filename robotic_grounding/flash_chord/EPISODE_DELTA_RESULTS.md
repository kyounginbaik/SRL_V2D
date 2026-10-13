# Episode delta diagnostic: 2026-10-13

One frozen 5,033,984-step policy and its starting reference; 2,048 candidates per batch, three upper-search seeds (43–45), zero VOC, one batch per seed. These are not three independently trained lower policies. All 12 jobs completed on L40S GPUs on node01.

| Variant | Parameters | Accepted | Gate-passing candidates per seed | Best valid batch cost improvement |
|---|---:|---:|---|---|
| episode_wrist | 12 | 1/3 | 40, 901, 68 | 11.1%, 19.8%, 7.3% |
| episode_full | 56 | 1/3 | 166, 151, 88 | 14.2%, 13.9%, 10.5% |
| chunk_wrist | 204 | 0/3 | 235, 193, 114 | 17.0%, 10.9%, 9.4% |
| chunk_full | 952 | 0/3 | 196, 181, 61 | 10.3%, 12.0%, 20.6% |

The cost improvements above are from the parallel batch, before the separate confirmation rollout. Lower cost does not guarantee publication. Rejection reasons for each selected candidate are in [the JSON results](episode_delta_results_20261013.json).

Both low-dimensional episode variants can produce accepted edits. The full-hand episode edit accepted for seed 43 changed each wrist by about 0.69 cm on average (0.70 cm maximum) and fingers by 1.68 degrees RMS. This is a useful small edit, not a successful task demonstration: its confirmation mean object position error remained about 9.4 cm.

High-dimensional searches also produced many eligible candidates in their batched evaluations. Thus this test does not support the claim that the large space prevents search from finding improving proposals altogether. Failures frequently occurred at confirmation through object/contact regression gates.

Unedited baseline cost varied from 16.121 to 18.086, despite identical checkpoint/reference hashes, deterministic policy inference, and a fixed evaluation reset seed. Even the four seed-43 baselines differ. We observed evaluation variability; its underlying numerical/physics cause was not isolated. This matters because contact and per-object position regression tolerances are 2 mm.

The experiment changes temporal flexibility as well as dimension. Three search seeds at one checkpoint cannot establish a causal dimensionality effect, persistent grasp feasibility, or end-to-end training superiority. It does establish that a shared full-hand episode delta is implemented and can pass the current gate. A sensible next check is repeated confirmation of shortlisted edits and baseline repeatability before drawing stronger conclusions or replacing the long-running training configuration.

Code validation: 21 targeted tests passed, one skipped; 26 FlashSAC learner tests passed; Python syntax and Git whitespace checks passed. A stale VOC test was aligned with the existing object-only gate (hand RMS remains diagnostic); runtime VOC behavior was not changed by that test correction.

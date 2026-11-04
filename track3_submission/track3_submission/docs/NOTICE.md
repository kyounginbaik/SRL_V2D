# Third-party code

| Path | Project | License |
|---|---|---|
| `third_party/DexAgent/` | DexAgent, github.com/dexagent1/DexAgent @ e29bdad | MIT (`third_party/DexAgent/LICENSE`) |
| `challenge_reconstruction/*/` that copy or call FlashCHORD, the V2D submission kit, or `video_to_data` | NVIDIA video_to_data / V2D submission kit | Apache-2.0 (code) and CC-BY-4.0 (docs), per upstream `LICENSE` |
| `challenge_reconstruction/reconstruction_recovery_20261026/vendor/` (not copied here; referenced by the recon streams) | point-to-pose, CoTracker and others | see each project's license |

Everything else in this overlay is SRL code, released under Apache-2.0 (`track3_submission/LICENSE`).
TODO before release: confirm the license of every vendored model and asset loaded at run time (SAM2/SAM3 weights, MANO, Sharpa hand assets).

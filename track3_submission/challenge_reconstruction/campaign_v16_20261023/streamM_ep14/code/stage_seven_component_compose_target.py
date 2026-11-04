#!/usr/bin/env python3
"""Wrapper around package_code/stage_seven_component.py (unchanged) for a component whose object target track was explicitly
corrected (episode 14 object reference re-targeted to the compose_eval poses, Stream M phase 6). Only the source-lineage check differs: instead of requiring the adapter's
object_poses.npz to be byte-identical to the user ZIP, it requires (a) the ZIP poses to equal the correction record's original
hash, (b) the adapter poses to equal the correction record's corrected hash, and records both plus the correction evidence.
The frozen runtime, portable replay and common evaluator are executed exactly as in the original staging script."""
import importlib.util, json, sys
from pathlib import Path
ORIG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/stage_seven_component.py')
spec = importlib.util.spec_from_file_location('stage_seven_component', ORIG); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
_original = S.provenance

def provenance(args, adapter, output):
    archive = S.read(args.base / 'input_manifest.json')
    source = Path(adapter['source']); targets = source / 'object_poses.npz'
    supplied = Path(archive['source']) / f'episode_{args.episode:06d}' / 'object_poses.npz'
    correction_path = Path(adapter['new_input_provenance']['hawor_source']) / 'compose_target_correction.json'
    corr = S.read(correction_path)
    if S.sha(supplied) != corr['original_object_poses_sha256']: raise ValueError('Correction record does not start from the user ZIP poses')
    if S.sha(targets) != corr['corrected_object_poses_sha256'] or S.sha(targets) != adapter['source_object_poses_sha256']:
        raise ValueError('Adapter poses differ from the recorded corrected track')
    real_sha = S.sha
    S.sha = lambda p: corr['original_object_poses_sha256'] if Path(p) == targets else real_sha(p)   # only the ZIP-equality comparison
    try:
        paths = _original(args, dict(adapter, source_object_poses_sha256=corr['original_object_poses_sha256']), output)
    finally:
        S.sha = real_sha
    rec = S.read(output); rec.update(original_object_targets_byte_identical=False, object_poses_sha256=real_sha(targets),
        user_zip_object_poses_sha256=real_sha(supplied), object_target_correction=dict(path=str(correction_path), sha256=real_sha(correction_path), record=corr),
        wrappers={**rec['wrappers'], str(Path(__file__).resolve()): real_sha(__file__)})
    S.write(output, rec)
    return [*paths, correction_path]

S.provenance = provenance
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--episode', type=int, required=True); p.add_argument('--base', type=Path, default=S.BASE); p.add_argument('--actor-report', type=Path)
    for k in ('checkpoint', 'training-report', 'evaluation', 'episode-dir', 'base-training-report', 'quality-dir', 'video'): p.add_argument('--' + k, type=Path)
    p.add_argument('--lineage-file', type=Path, action='append', default=[])
    S.run(p.parse_args())

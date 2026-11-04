#!/usr/bin/env python3
"""Stream S (episode 15, NEW spoon track) copy of rackscan_20261021/ep30_v8/code/stage_seven_component_corrected_track.py: wrapper around the unchanged
package_code/stage_seven_component.py for a component whose object target track was explicitly corrected (episode 15 wooden_spoon re-targeted to the NEW compose_eval_20261021 track,
streamS_ep15/code/retarget_newtrack.py). Only the source-lineage check differs: instead of requiring the adapter's object_poses.npz to be
byte-identical to the user ZIP, it requires (a) the ZIP poses to equal the correction record's original hash, (b) the adapter poses to
equal the record's corrected hash, and records both plus the correction evidence. Changes vs the ep30 wrapper: the record is read from a fixed path in the stream dir
(newtrack/spoon_track_correction.json) and the adapter's hawor-source poses must hash to its corrected_object_poses_sha256."""
import importlib.util, json, sys
from pathlib import Path
ORIG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/stage_seven_component.py')
spec = importlib.util.spec_from_file_location('stage_seven_component', ORIG); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
_original = S.provenance
RECORD = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/newtrack/spoon_track_correction.json')

def provenance(args, adapter, output):
    archive = S.read(args.base / 'input_manifest.json')
    source = Path(adapter['source']); targets = source / 'object_poses.npz'
    supplied = Path(archive['source']) / f'episode_{args.episode:06d}' / 'object_poses.npz'
    correction_path = RECORD
    corr = S.read(correction_path)
    hs = Path(adapter['new_input_provenance']['hawor_source'])
    if S.sha(hs / 'object_poses.npz') != corr['corrected_object_poses_sha256']: raise ValueError('adapter hawor source poses differ from the corrected track')
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
        user_zip_object_poses_sha256=real_sha(supplied), object_target_correction=dict(path=str(correction_path), sha256=real_sha(correction_path),
        record={k: v for k, v in corr.items() if k != 'per_frame'}),
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

"""Stage the selected unrounded episode-1 actor; no shared runtime edits."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
from prepare_shared_component import run

B = Path(__file__).resolve().parents[1]
R = B / 'refinement_20261018/episode_000001/reconstruction_repair'
P = R / 'patch_height094'
S = P / 'learned_carry_refinement_v2'
O = B / 'refinement_20261018/shared_policy_handoff/episode_000001'


def main():
    lineage = [P/'learned_hand_margins/search_v3/training_report.json',
        P/'learned_hand_margins/search_v3/settings.json', S/'settings.json',
        S/'final_validation.json', R/'status.json',
        R/'pot_geometry_rgb_depth_height094/fit.json',
        R/'pot_geometry_rgb_depth_v3/holdout_audit.json',
        P/'pickup_4s/initial_audit.json', P/'repaired_adapter/mesh_export_audit.json',
        S/'evaluation/audit/alignment_bias.json', R/'source_pose_registration_v1/review.json']
    for path in lineage:
        if not path.is_file():
            raise FileNotFoundError(path)
    run(SimpleNamespace(run=S/'evaluation', checkpoint=S/'policy.npz',
        training_report=S/'training_report.json',
        base_training_report=P/'learned_feedback/centers_1024_gain_0.03/training/training_report.json',
        episode_dir=P/'repaired_adapter', quality_dir=S/'evaluation/audit',
        output=O, lineage_file=lineage))
    shutil.copytree(S/'videos', O/'videos')
    shutil.copyfile(S/'final_validation.json', O/'validation/cross_host_validation.json')
    shutil.copyfile(P/'repaired_adapter/mesh_export_audit.json', O/'provenance/repaired_mesh_export_audit.json')
    shutil.copyfile(R/'status.json', O/'provenance/reconstruction_evidence.json')
    # Explicit source evidence and selected command precision are retained for review.
    evidence=O/'provenance/reconstruction';evidence.mkdir()
    for path in [R/'pot_geometry_rgb_depth_height094/fit.json',
                 R/'pot_geometry_rgb_depth_v3/holdout_audit.json',
                 P/'pickup_4s/initial_audit.json', P/'pickup_4s/settled_rgb_overlay.jpg',
                 S/'evaluation/audit/alignment_bias.json']:
        shutil.copyfile(path,evidence/path.name)
    print(json.dumps(dict(component=str(O),runtime_files_changed=False)))


if __name__=='__main__':
    main()

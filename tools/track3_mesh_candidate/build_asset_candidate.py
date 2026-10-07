"""Build a Track 3 mesh candidate from released assets and a video-derived spoon.

This is an asset-retrieval baseline for known object identities. The source
manifest makes the distinction from video-only reconstruction explicit.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get('V2D_MESH_CANDIDATE_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/track3_asset_candidate'))
PUBLIC = Path('/data1/video_to_data_challenge/track_3/public')
KIT = Path('/data1/video_to_data_challenge/v2d_submission_kit')
DEFAULT_SPOON = Path('/data1/SRL_V2D/challenge_reconstruction/spoon_mesh_candidates/consensus_alpha50.glb')


def public_visual_mesh(name):
    directory = PUBLIC / 'mesh' / name
    paths = [p for p in directory.glob('*.glb') if not p.name.endswith('_collision.glb')]
    if len(paths) != 1:
        raise ValueError(f'{name}: expected one public visual mesh, found {paths}')
    return paths[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spoon-mesh', type=Path, default=DEFAULT_SPOON)
    args = parser.parse_args()
    roster = json.loads((KIT / 'data/track_3_evaluation_objects.json').read_text())['episodes']
    entries = []
    for episode_text, names in sorted(roster.items(), key=lambda kv: int(kv[0])):
        episode = int(episode_text)
        for slot, name in enumerate(names):
            if name == 'wooden_spoon':
                source, source_kind = args.spoon_mesh, 'evaluation_video_reconstruction'
            else:
                source, source_kind = public_visual_mesh(name), 'released_public_asset'
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            target = ROOT / 'meshes' / f'episode_{episode:06d}' / f'{name}.glb'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entries.append({'episode': episode, 'object_slot': slot, 'object_name': name,
                            'source_kind': source_kind, 'source': str(source), 'mesh': str(target)})
    if len(entries) != 36:
        raise ValueError(f'expected 36 episode-object meshes, found {len(entries)}')
    (ROOT / 'manifest.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(f'Prepared {len(entries)} meshes at {ROOT / "meshes"}')


if __name__ == '__main__':
    main()

"""Prepare video-only image/mask inputs for the public Track 3 mesh run."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public')


def main():
    for item in json.loads((ROOT / 'objects.json').read_text()):
        episode = item['episode']
        frame = item['frame']
        source = SOURCE / f'episode_{episode:06d}'
        assert (source / f'frames/{frame:06d}.png').is_file(), item
        out = ROOT / 'inputs' / item['name']
        out.mkdir(parents=True, exist_ok=True)
        if item['mask'] == 'sam2':
            x0, y0, x1, y1 = item['box']
            prompt = {'prompts': [{'frame_index': frame, 'object_id': 1,
                                   'points': None, 'point_labels': None,
                                   'box': {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1},
                                   'mask_path': None}]}
            (out / 'sam2_prompt.json').write_text(json.dumps(prompt, indent=2) + '\n')
        else:
            assert (source / f'masks/1/{frame:06d}.png').is_file(), item
        print(item['name'], source / f'frames/{frame:06d}.png', item['mask'])


if __name__ == '__main__':
    main()

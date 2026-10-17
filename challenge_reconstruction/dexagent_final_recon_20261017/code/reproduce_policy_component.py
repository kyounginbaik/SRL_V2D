#!/usr/bin/env python3
"""Build and replay an authentic learned-controller bundle with explicit relocation.

The original checkpoint, XML, measured teacher data and source object targets are
preserved byte for byte. A copied checkpoint changes ONLY the XML hash after all
compiled model arrays/options match. No policy weights or physics are edited.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

# Match the recorded online evaluator. Multithreaded BLAS can change reductions
# by 1e-16, which can amplify in contact-rich dynamics; set before NumPy imports.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import mujoco
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def copy(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha(dst) != sha(src):
        raise FileExistsError(f'Refusing to replace different bytes: {dst}')
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dict(source=str(src.resolve()), path=str(dst), sha256=sha(dst))


def fingerprint(model):
    # Only these arrays/scalars encode filenames rather than compiled physics.
    # nbuffer is allocated byte count and includes the variable-length path table.
    excluded = {'paths', 'npaths', 'nbuffer', 'mesh_pathadr', 'tex_pathadr', 'hfield_pathadr'}
    result = {}
    for prefix, obj in [('model', model), ('option', model.opt)]:
        for key in dir(obj):
            if key.startswith('_') or (prefix == 'model' and key in excluded):
                continue
            value = getattr(obj, key)
            if isinstance(value, np.ndarray):
                result[prefix+'.'+key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
            elif isinstance(value, (int, float, str)):
                result[prefix+'.'+key] = value
    return dict(checked=result, excluded_path_metadata=sorted(excluded), mujoco_version=mujoco.__version__)


def build(args):
    b = Path(__file__).resolve().parents[1]
    dest = args.bundle.resolve()
    if (dest/'bundle_manifest.json').exists():
        raise FileExistsError(dest/'bundle_manifest.json')
    report = read(args.evaluation/'result.json')
    training = read(args.training_report)
    base_training = read(args.base_training_report or args.training_report)
    adapter = args.episode_dir or Path(report.get('input_dir') or b/'episodes'/report['episode'])
    assert report['learned_policy'] and report['reference_proxy_success']
    assert report['execution']['independent_control_replay']['ok']
    assert report['policy_checkpoint_sha256'] == sha(args.checkpoint)
    assert training['checkpoint_sha256'] == report['policy_checkpoint_sha256']
    episode = report['episode']
    manifest = read(adapter/'input_manifest.json')
    source = Path(manifest['source'])
    geometry = read(adapter/'metric_geometry.json')
    original_scene = Path(report['execution']['scene'])
    inventory = []
    for src, rel in [(original_scene,'original/scene_physics.xml'),
                     (args.checkpoint,'original/policy.npz'),
                     (args.training_report,'original/training_report.json'),
                     (args.evaluation/'result.json','original/evaluation_result.json'),
                     (Path(report['execution']['trajectory']),'original/evaluation_trajectory.npz'),
                     (Path(base_training['demonstration']),'original/teacher_trajectory.npz'),
                     (adapter/'input_manifest.json','original/input_manifest.json'),
                     (adapter/'metric_geometry.json','original/metric_geometry.json')]:
        inventory.append(copy(src,dest/rel))
    inventory.append(copy(args.base_training_report or args.training_report,dest/'original/base_training_report.json'))
    for index, src in enumerate(args.lineage_file):
        inventory.append(copy(src,dest/'lineage'/f'{index:03d}_{src.name}'))
    if sha(Path(base_training['scene'])) != sha(original_scene):
        raise ValueError('Teacher and learned evaluation scenes differ')
    for name in ['manifest.json','object_poses.npz','reference.parquet']:
        inventory.append(copy(source/name,dest/'source'/episode/name))
    for name, obj in geometry['objects'].items():
        inventory.append(copy(obj['mesh_path'],dest/'source'/episode/'assets'/Path(obj['mesh_path']).name))
    for name in ['learned_feedback_policy.py','component_policy.py','evaluate_policy_component.py',
                 'posture_feedback_policy.py','posture_feedback_policy_v3.py','posture_feedback_policy_v4.py',
                 'joint_margin_policy.py','pinky_margin_policy_v2.py','release_feedback_policy.py','wrist_reference.py',
                 'floating_runtime.py','evaluate_mp_sr.py','grasp_stability.py','reproduce_policy_component.py']:
        inventory.append(copy(b/'code'/name,dest/'code'/name))
    dex = b.parents[1]/'third_party/DexAgent'
    for src in sorted((dex/'tool_library').rglob('*.py')):
        inventory.append(copy(src,dest/'third_party/DexAgent'/src.relative_to(dex)))
    inventory.append(copy(dex/'LICENSE',dest/'third_party/DexAgent/LICENSE'))
    tree = ET.parse(original_scene)
    if tree.findall('.//include'):
        raise ValueError('Expected frozen include-free scene XML')
    assets = []
    for index, element in enumerate(tree.iter()):
        value = element.get('file')
        if value is None:
            continue
        asset = Path(value)
        if not asset.is_absolute():
            raise ValueError('Expected frozen absolute source asset path')
        relative = Path('assets')/(sha(asset)+asset.suffix)
        inventory.append(copy(asset,dest/relative))
        assets.append(dict(element_index=index,source=str(asset),path=str(relative),sha256=sha(asset)))
    compiled = fingerprint(mujoco.MjModel.from_xml_path(str(original_scene)))
    write(dest/'original/compiled_model_fingerprint.json',compiled)
    packages = ['mujoco','numpy','scipy','trimesh','pyarrow','pandas','fast-simplification']
    versions = {p:importlib.metadata.version(p) for p in packages}
    (dest/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    bundle = dict(policy_architecture=report['policy_architecture'],episode=episode,episode_id=int(episode.split('_')[-1]),assets=assets,
        source_reference_sha256=manifest['source_reference_sha256'],
        source_object_poses_sha256=manifest['source_object_poses_sha256'],
        original_checkpoint_sha256=report['policy_checkpoint_sha256'],
        original_scene_sha256=sha(original_scene),compiled_fingerprint_sha256=sha(dest/'original/compiled_model_fingerprint.json'),
        inventory=[dict(x,path=str(Path(x['path']).relative_to(dest))) for x in inventory],
        expected_reference_proxy_mp_sr=1.,python_version=sys.version,requirements=versions,
        scope='One learned-policy episode component, not the complete 20-episode submission or official hidden-GT score',
        source_targets_immutable=True,inference_threads=1,
        relocation_changes='Only asset paths, source-manifest paths and copied checkpoint scene_sha256 metadata',
        weights_change=False,physics_change=False,uploaded=False)
    write(dest/'bundle_manifest.json',bundle)
    (dest/'README.md').write_text('''# Learned-policy episode reproduction

This bundle preserves the original learned checkpoint, physical teacher data,
fresh evaluation recording and reconstructed object targets. It is one episode
component, not a complete Track 3 submission. MP-SR uses reconstructed references
as a proxy for unavailable official ground truth.

Create a Python 3.11 environment and install `requirements.txt`, then run:

```sh
python code/reproduce_policy_component.py replay --bundle . --output /tmp/policy-reproduction
```

The helper verifies every bundled file, recreates only absolute asset paths,
checks all compiled model arrays/options against the original fingerprint,
and changes only `scene_sha256` in a copied checkpoint. It then performs fresh
online policy inference, independent recorded-control replay and proxy scoring.
The final report compares all recorded physical states/actions with the original.
No reference commands or future object targets are given to the policy.
The launcher fixes BLAS/OpenMP to one thread, matching the recorded evaluator;
contact-rich trajectories can amplify even tiny floating-point differences.

For an unmodified BC actor, add `--retrain` to reproduce its fitting. For a
physically fine-tuned actor, only full-checkpoint replay is automated; its recorded
training/search lineage is preserved separately. Base BC fitting uses
the unchanged measured teacher states/actions, explicitly rebinding only its
scene hash to the equivalent relocated XML. The new policy is separately saved;
the primary evaluation always uses the preserved learned weights.

The policy uses radial-basis behavior cloning with locally fitted proprioceptive
feedback and may include a learned posture head and physically optimized bounded
correction parameters. `bundle_manifest.json` identifies its exact architecture.
It is not FlashSAC. Synthetic recovery labels assume a local stabilizing
teacher; they are not claimed successful physical rollouts. Interleaved held-out
samples measure interpolation, not task generalization. Custom MuJoCo execution
does not claim bundled official-harness certification. Original archive and code
provenance remain available in the surrounding episode handoff.
''')
    print(json.dumps({'bundle':str(dest),'inventory_files':len(inventory),'scene_assets':len(assets)}))


def replay(args):
    bundle, output = args.bundle.resolve(),args.output.resolve()
    if (output/'portable_validation.json').exists():
        raise FileExistsError(output/'portable_validation.json')
    meta = read(bundle/'bundle_manifest.json')
    for item in meta['inventory']:
        if sha(bundle/item['path']) != item['sha256']:
            raise ValueError(f'Bundle file changed: {item["path"]}')
    if sha(bundle/'original/compiled_model_fingerprint.json') != meta['compiled_fingerprint_sha256']:
        raise ValueError('Compiled model fingerprint changed')
    output.mkdir(parents=True,exist_ok=True)
    tree = ET.parse(bundle/'original/scene_physics.xml')
    elements = list(tree.iter())
    for item in meta['assets']:
        asset = bundle/item['path']
        if sha(asset) != item['sha256']:
            raise ValueError('Asset changed')
        elements[item['element_index']].set('file',str(asset))
    scene = output/'scene_relocated.xml'
    tree.write(scene,encoding='unicode')
    model_fingerprint = fingerprint(mujoco.MjModel.from_xml_path(str(scene)))
    expected = read(bundle/'original/compiled_model_fingerprint.json')
    if model_fingerprint != expected:
        keys = sorted(set(model_fingerprint['checked'])|set(expected['checked']))
        difference = [k for k in keys if model_fingerprint['checked'].get(k)!=expected['checked'].get(k)]
        write(output/'model_difference.json',dict(different_fields=difference,actual=model_fingerprint,expected=expected))
        raise ValueError(f'Relocated compiled model differs: {difference}')
    with np.load(bundle/'original/policy.npz',allow_pickle=False) as z:
        checkpoint = {k:z[k] for k in z.files}
    original_scene_hash = str(checkpoint['scene_sha256'])
    if original_scene_hash != meta['original_scene_sha256']:
        raise ValueError('Original checkpoint/scene mismatch')
    checkpoint['scene_sha256'] = np.array(sha(scene))
    checkpoint_path = output/'policy_relocated.npz'
    np.savez_compressed(checkpoint_path,**checkpoint)
    with np.load(bundle/'original/policy.npz',allow_pickle=False) as original, np.load(checkpoint_path,allow_pickle=False) as relocated:
        identical_weights = all(np.array_equal(original[k],relocated[k]) for k in original.files if k!='scene_sha256')
    if not identical_weights:
        raise ValueError('Relocation changed learned weights')
    # Keep metadata required by the evaluator outside the immutable bundle.
    episode = meta['episode']
    runtime = output/'runtime'
    manifest = read(bundle/'original/input_manifest.json')
    manifest['source'] = str(bundle/'source'/episode)
    manifest['source_reference'] = str(bundle/'source'/episode/'reference.parquet')
    write(runtime/'episodes'/episode/'input_manifest.json',manifest)
    geometry = read(bundle/'original/metric_geometry.json')
    for obj in geometry['objects'].values():
        obj['mesh_path'] = str(bundle/'source'/episode/'assets'/Path(obj['mesh_path']).name)
    write(runtime/'episodes'/episode/'metric_geometry.json',geometry)
    sys.path[:0] = [str(bundle/'code'),str(bundle/'third_party/DexAgent')]
    import evaluate_policy_component as evaluator
    evaluator.B = runtime
    result = evaluator.evaluate(meta['episode_id'],scene,checkpoint_path,output/'evaluation',str(runtime/'episodes'/episode))
    compare = {}
    with np.load(bundle/'original/evaluation_trajectory.npz',allow_pickle=False) as expected_trajectory, np.load(result['execution']['trajectory'],allow_pickle=False) as actual:
        for key in ['qpos','qvel','ctrl','time_s','substep_qpos','substep_qvel','substep_ctrl','substep_time_s',
                    'initial_qpos','initial_qvel','initial_state','final_state']:
            compare[key] = dict(exact=bool(np.array_equal(expected_trajectory[key],actual[key])),
                max_abs_difference=float(np.max(np.abs(expected_trajectory[key]-actual[key]),initial=0)))
    retrained = None
    if args.retrain:
        if meta['policy_architecture'] != 'learned_feedback_policy.LearnedPolicy':
            raise ValueError('This actor includes physical fine-tuning; replay is supported, but --retrain would only reproduce its base BC stage. Preserve and use the recorded lineage instead.')
        from learned_feedback_policy import train
        with np.load(bundle/'original/teacher_trajectory.npz',allow_pickle=False) as z:
            teacher = {k:z[k] for k in z.files}
        teacher['scene_sha256'] = np.array(sha(scene))
        relocated_teacher = output/'teacher_relocated_metadata.npz'
        np.savez_compressed(relocated_teacher,**teacher)
        settings = read(bundle/'original/base_training_report.json')
        retrained = train(scene,relocated_teacher,output/'retraining',
            centers=settings['phase_basis_count'],feedback_gain=settings['feedback_teacher_position_gain'],
            velocity_gain=settings['feedback_teacher_velocity_gain'],seed=settings['seed'])
        with np.load(retrained['checkpoint'],allow_pickle=False) as z:
            retrained['weight_arrays_exact'] = all(np.array_equal(z[k],checkpoint[k]) for k in z.files if k!='scene_sha256')
    validation = dict(episode=episode,bundle=str(bundle),working_directory=str(Path.cwd()),
        original_scene_sha256=original_scene_hash,relocated_scene_sha256=sha(scene),
        original_checkpoint_sha256=meta['original_checkpoint_sha256'],relocated_checkpoint_sha256=sha(checkpoint_path),
        original_weights_unchanged=identical_weights,compiled_model_arrays_options_exact=True,
        excluded_model_source_path_metadata=expected['excluded_path_metadata'],
        physical_trajectory_comparison=compare,all_physical_states_actions_exact=all(v['exact'] for v in compare.values()),
        source_object_targets_unchanged=sha(bundle/'source'/episode/'object_poses.npz')==meta['source_object_poses_sha256'],
        reference_proxy_mp_sr=result['metrics']['reference_proxy_mp_sr'],
        independent_control_replay=result['execution']['independent_control_replay'],retraining=retrained,
        complete_submission_ready=False,uploaded=False)
    write(output/'portable_validation.json',validation)
    if not validation['all_physical_states_actions_exact']:
        raise RuntimeError('Portable fresh rollout differs; review report before claiming reproduction')
    print(json.dumps({'portable_validation':str(output/'portable_validation.json'),
        'all_physical_states_actions_exact':validation['all_physical_states_actions_exact'],
        'reference_proxy_mp_sr':validation['reference_proxy_mp_sr']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    build_parser=sub.add_parser('build');build_parser.add_argument('--bundle',type=Path,required=True)
    build_parser.add_argument('--evaluation',type=Path,required=True)
    build_parser.add_argument('--checkpoint',type=Path,required=True)
    build_parser.add_argument('--training-report',type=Path,required=True)
    build_parser.add_argument('--base-training-report',type=Path)
    build_parser.add_argument('--episode-dir',type=Path)
    build_parser.add_argument('--lineage-file',type=Path,action='append',default=[])
    replay_parser=sub.add_parser('replay');replay_parser.add_argument('--bundle',type=Path,required=True)
    replay_parser.add_argument('--output',type=Path,required=True);replay_parser.add_argument('--retrain',action='store_true')
    args=p.parse_args()
    build(args) if args.mode=='build' else replay(args)

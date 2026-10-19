"""Pack one real successful rollout with actual meshes using the latest public kit."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import numpy as np


def pack_component(destination, poses, names, mesh_paths, manifest, kit, code_commit_url=''):
    # These wheels are isolated to postprocessing; never change training's NumPy/runtime.
    vendor=Path(__file__).resolve().parents[1]/'reproduction/packaging_python'
    if vendor.is_dir():sys.path.insert(0,str(vendor))
    import pandas as pd
    kit=Path(kit)
    sys.path.insert(0,str(kit))
    from v2dlb.track3_pose import pack_object_meshes, validate_packed_meshes
    from v2dlb.submission_parquet import read_trajectory, provenance
    destination=Path(destination)
    raw=read_trajectory(destination.parent/(manifest['episode']+'.parquet'))
    prov=provenance(raw)
    ep=int(manifest['episode'].split('_')[-1]);t,o=poses.shape[:2]
    rows=[f'track_3/e{ep:06d}/w000/f{f:06d}/o{j}' for f in range(t) for j in range(o)]
    sample=pd.read_parquet(kit/'data/track_3_sample_submission.parquet')
    expected=sample[sample.row_id.str.startswith(f'track_3/e{ep:06d}/')].row_id.tolist()
    if rows!=expected:raise ValueError('Actual trajectory rows differ from exact public episode grid')
    columns=['pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w']
    frame=pd.DataFrame(poses.reshape(-1,7),columns=columns);frame.insert(0,'row_id',rows)
    for key in ('eval_script_sha256','eval_code_sha256','checkpoint_sha256'):
        frame[key]=prov[key]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frame['packer_script_sha256']=digest(__file__)
    frame['uploader_script_sha256']=digest(kit/'submit/v2d_submit.py')
    frame['code_commit_url']=code_commit_url
    roster=json.loads((kit/'data/track_3_evaluation_objects.json').read_text())['episodes']
    if roster[str(ep)]!=names:raise ValueError('Mesh roster/order mismatch')
    with tempfile.TemporaryDirectory(prefix='.mesh-pack-',dir=destination.parent) as temporary:
        root=Path(temporary)/manifest['episode'];root.mkdir()
        for name,source in zip(names,mesh_paths,strict=True):
            source=Path(source)
            shutil.copy2(source,root/(name+source.suffix))
        frame=pack_object_meshes(frame,Path(temporary),roster)
        validate_packed_meshes(frame)
        frame.to_parquet(destination,index=False)
    loaded=pd.read_parquet(destination)
    validate_packed_meshes(loaded)
    np.testing.assert_allclose(loaded[columns].to_numpy(),poses.reshape(-1,7),rtol=0,atol=1e-7)
    return {'combined_parquet':destination.name,'rows':len(frame),'objects':names,
            'mesh_payloads':int((frame.object_mesh_b64!='').sum()),
            'schema_and_public_episode_grid_valid':True,
            'code_commit_url':code_commit_url,
            'complete_official_submission':False,
            'ready_component':bool(code_commit_url),
            'missing_provenance':[] if code_commit_url else ['Published code_commit_url'],
            'official_pinned_evaluator_compliance_verified':False,
            'mesh_coordinate_policy':'Original object mesh coordinates; no centering/rescaling. Official fixed vertex/face budgeting only.'}

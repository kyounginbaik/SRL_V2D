from pathlib import Path
import json
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation
import archive_success as mod


def poses(t=4,w=1):
    ref=np.zeros((t,2,7));ref[...,6]=1.;ref[:,1,0]=.2
    return np.broadcast_to(ref[:,None],(t,w,2,7)).copy(),ref


def test_mean_not_per_frame_and_every_object():
    a,r=poses();a[1,0,1,0]+=.08
    assert mod.score_worlds(a,r,['a','b'])[0]['passed']
    a[:,:,1,0]+=.04
    assert not mod.score_worlds(a,r,['a','b'])[0]['passed']


def test_strict_position_and_orientation_thresholds():
    a,r=poses();a[:,:,1,0]+=.030001
    assert not mod.score_worlds(a,r,['a','b'])[0]['passed']
    a,r=poses();a[:,:,1,3:]=Rotation.from_euler('z',30.001,degrees=True).as_quat()
    assert not mod.score_worlds(a,r,['a','b'])[0]['passed']


def test_invalid_world_does_not_become_success():
    a,r=poses(w=3);a[2,0,0,0]=np.nan;a[1,1,0,3:]=0
    scores=mod.score_worlds(a,r,['a','b'])
    assert [s['passed'] for s in scores]==[False,False,True]


def test_single_rigid_alignment():
    a,r=poses();rotation=Rotation.from_euler('xyz',[.4,.2,.7]);a[:,0,:,:3]=rotation.apply(r[...,:3].reshape(-1,3)).reshape(r.shape[:2]+(3,))+[1,2,3]
    a[:,0,:,3:]=(rotation*Rotation.from_quat(r[...,3:].reshape(-1,4))).as_quat().reshape(r.shape[:2]+(4,))
    assert mod.score_worlds(a,r,['a','b'])[0]['passed']


def test_symmetry_math_is_opt_in_no_axis_guessing():
    q=Rotation.from_rotvec(np.array([[0,0,np.pi/2],[0,0,np.pi]])).as_quat();r=np.tile([0,0,0,1],(2,1))
    assert np.all(mod.orientation_error(q,r)>.5)
    assert np.max(mod.orientation_error(q,r,{'type':'continuous','axis':[0,0,1]}))<1e-7
    assert mod.orientation_error(q[1:],r[1:],{'type':'discrete','quaternions_xyzw':[[0,0,1,0]]})[0]<1e-7


@pytest.mark.parametrize('both_cups',[False,True])
def test_symmetric_scene_alignment_does_not_use_unobservable_cup_yaw(both_cups):
    a,r=poses();a[:,:,0,3:]=Rotation.from_rotvec([0,0,np.pi/2]).as_quat()
    symmetry={'a':{'type':'continuous','axis':[0,0,1]}}
    if both_cups:
        symmetry['b']={'type':'continuous','axis':[0,0,1]}
        a[:,:,1,3:]=Rotation.from_rotvec([0,0,-np.pi/2]).as_quat()
    assert mod.score_worlds(a,r,['a','b'],symmetry)[0]['passed']


@pytest.fixture
def fixture(tmp_path):
    base=Path(mod.__file__).resolve().parents[1]
    real=base/'frozen'/base.name
    original=pq.read_table(real/'reference.parquet').to_pylist()[0]
    t=len(original['object_body_position']);a,r=poses(t=t,w=2)
    a[:,0,1,0]+=.05;a[:,1,1,0]+=.002
    root=tmp_path/base.name;root.mkdir();shutil.copytree(real/'assets',root/'assets')
    names=original['safe_object_body_names']
    row={'object_body_position':r[...,:3].tolist(),'object_body_wxyz':r[...,[6,3,4,5]].tolist(),
         'safe_object_body_names':names,'fps':20.,'object_mesh_paths':[str(root/'assets'/Path(p).name) for p in original['object_mesh_paths']]}
    source=root/'reference.parquet';pq.write_table(pa.Table.from_pylist([row]),source)
    checkpoint=root/'policy.safetensors';checkpoint.write_bytes(b'unit-test checkpoint fixture')
    trajectory=root/'rollout.npz'
    np.savez(trajectory,actual=a,reference=r,frame_indices=np.arange(t),object_names=names,fps=20.)
    manifest={'episode':base.name,'object_names':names,'fps':20.,'frames':t,'world_count':2,'world_seeds':[44,44],
      'checkpoint_path':str(checkpoint),'checkpoint_sha256':mod.digest(checkpoint),
      'source_reference_path':str(source),'source_reference_sha256':mod.digest(source),
      'evaluation_script_path':str(base/'code/render_parallel_checkpoint.py'),
      'voc_scale':0,'voc_force_limit':0,'voc_torque_limit':0,'reset_count_after_start':0,
      'reset_assist_steps':0,'full_horizon':True,'object_pose_forcing':False,'auto_reset':False,'environment_steps':123}
    path=root/'manifest.json';path.write_text(json.dumps(manifest))
    return root,trajectory,path,manifest,a,r


def test_archive_idempotent_actual_world_export_and_partial_readiness(fixture):
    root,trajectory,path,m,a,r=fixture
    result=mod.archive(trajectory,path,root/'archive')
    assert result['any_success'] and len(result['archives'])==1
    success=Path(result['archives'][0]);first=(root/'archive/first_success.json').read_bytes()
    hashes=(success/'sha256.json').read_bytes()
    result2=mod.archive(trajectory,path,root/'archive')
    assert result2['archives']==result['archives']
    assert (root/'archive/first_success.json').read_bytes()==first
    assert (success/'sha256.json').read_bytes()==hashes
    info=json.loads((success/'success.json').read_text())
    assert info['selected_original_world_id']==1 and info['submitted_world_id']==0
    assert info['official_heldout_success'] is None and info['cd_o'] is None
    assert not info['complete_official_submission']
    assert info['schema_component_ready'],info['packaging']
    table=pq.read_table(success/(m['episode']+'.parquet'))
    actual=np.stack([table['achieved_'+k].to_numpy() for k in ['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']],axis=-1)
    np.testing.assert_allclose(actual,a[:,1].reshape(-1,7),atol=1e-7)
    assert set(table['environment_id'].to_pylist())=={0}


@pytest.mark.parametrize('bad', ['offset','missing','reference','reset','assistance','checkpoint'])
def test_reject_wrong_frames_reference_assistance_or_provenance(fixture,bad):
    root,trajectory,path,m,a,r=fixture;t=len(r);frames=np.arange(t)
    if bad=='offset':frames+=1
    if bad=='missing':a=a[:-1];r=r[:-1];frames=frames[:-1]
    if bad=='reference':r[1,1,0]+=.01
    if bad=='reset':m['reset_count_after_start']=1
    if bad=='assistance':m['voc_force_limit']=1
    if bad=='checkpoint':m['checkpoint_sha256']='0'*64
    np.savez(trajectory,actual=a,reference=r,frame_indices=frames)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):mod.archive(trajectory,path,root/'archive')


def test_nonsuccess_is_normal_result(fixture):
    root,trajectory,path,m,a,r=fixture;a[:,:,1,0]+=.2
    np.savez(trajectory,actual=a,reference=r,frame_indices=np.arange(len(r)))
    result=mod.archive(trajectory,path,root/'archive')
    assert not result['any_success'] and result['archives']==[]
    assert not (root/'archive/first_success.json').exists()

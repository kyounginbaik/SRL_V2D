"""Missing hand detections are not pose labels; full reference time stays intact."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from flash_chord.data import mano_sharpa
from flash_chord.data.reference import hand_validity


def cells():
    result = dict(fps=20., source_dataset="test", object_name="object",
        object_body_names=["object"], object_mesh_paths=["object.obj"],
        object_mesh_radius=[.1], object_body_position=np.zeros((3,1,3)),
        object_body_wxyz=np.tile([1.,0.,0.,0.],(3,1,1)))
    for side in ("left","right"):
        result.update({f"robot_{side}_wrist_position": np.arange(9,dtype=float).reshape(3,3),
            f"robot_{side}_wrist_wxyz": np.tile([1.,0.,0.,0.],(3,1)),
            f"robot_{side}_finger_joints": np.arange(3,dtype=float).reshape(3,1),
            f"{side}_robot_finger_joint_names": [side+"_joint"],
            f"mano_{side}_object_contact_positions": np.zeros((3,1,3)),
            f"mano_{side}_object_contact_normals": np.zeros((3,1,3)),
            f"mano_{side}_object_contact_part_ids": np.zeros((3,1)),
            f"robot_{side}_frames": np.tile([0.,0.,0.,1.,0.,0.,0.],(3,1,1)),
            f"{side}_robot_frame_names": [side+"_tip"]})
    return result


def load_case(monkeypatch,tmp_path,row,**kwargs):
    monkeypatch.setattr(mano_sharpa,"read_parquet_row",lambda path:NS(path=tmp_path/"reference.parquet",cells=row))
    monkeypatch.setattr(mano_sharpa,"resolve_object_asset_specs",lambda **kw:())
    return mano_sharpa.load_mano_sharpa(tmp_path/"reference.parquet",**kwargs)


def test_invalid_finite_held_targets_preserve_trajectory_and_side_order(monkeypatch,tmp_path):
    row=cells();row["mano_left_valid"]=[True,False,True]
    reference=load_case(monkeypatch,tmp_path,row)
    assert reference.num_frames==3
    np.testing.assert_array_equal(reference.wrist_pos_w("left"),row["robot_left_wrist_position"])
    np.testing.assert_array_equal(reference.finger_joint_pos("left"),row["robot_left_finger_joints"])
    np.testing.assert_array_equal(hand_validity(reference,("right","left")),[[1,1],[1,0],[1,1]])
    window=reference.frame_window(1,3)
    np.testing.assert_array_equal(window.hand_valid("left"),[False,True])
    assert window.num_frames==2


def test_resampling_excludes_interpolations_with_unobserved_endpoint(monkeypatch,tmp_path):
    row=cells();row["mano_left_valid"]=[1,0,1]
    reference=load_case(monkeypatch,tmp_path,row,control_fps=40.)
    assert reference.num_frames==5
    np.testing.assert_array_equal(reference.hand_valid("left"),[1,0,0,0,1])
    np.testing.assert_array_equal(reference.hand_valid("right"),np.ones(5,bool))
    np.testing.assert_allclose(reference.wrist_pos_w("left")[:,0],[0.,1.5,3.,4.5,6.])


def test_legacy_references_without_masks_are_all_valid(monkeypatch,tmp_path):
    reference=load_case(monkeypatch,tmp_path,cells())
    np.testing.assert_array_equal(hand_validity(reference,("left","right")),np.ones((3,2),bool))
    np.testing.assert_array_equal(hand_validity(NS(num_frames=3),("right",)),np.ones((3,1),bool))


@pytest.mark.parametrize("mask",[[1,0],[1,2,0],[1,float("nan"),0],[[1],[0],[1]],[]])
def test_malformed_present_masks_fail(monkeypatch,tmp_path,mask):
    row=cells();row["mano_left_valid"]=mask
    with pytest.raises(ValueError,match="mano_left_valid"):
        load_case(monkeypatch,tmp_path,row)

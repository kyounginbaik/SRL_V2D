"""Adapt the fitted table normal to DexAgent's registered camera convention."""
import math
from tool_library.skills.fit_camera_to_table_plane import fit_camera_to_table_plane

TOOL = {
    "name": "fit_camera_to_table_plane_cv", "title": "Fit table camera in CV convention",
    "type": "skill", "category": "Perception & Grounding", "status": "tested",
    "summary": "Fit real metric depth and convert its normal to register_table_camera's positive downward pitch convention.",
    "description": "Calls the existing depth-plane fitter unchanged, then derives pitch and roll directly from its camera-facing world-up normal. Preserves the original angles as diagnostics. Verified against three synthetic depth planes generated from common.rigcam.R_cv_from_angles, including both roll signs. Does not change depth, height, normal or fitting thresholds.",
    "inputs": [
        {"name":"depth_npz","type":"path","description":"Depths in metres, depths array."},
        {"name":"masks_dir","type":"path","description":"Palette frame masks."},
        {"name":"focal_px","type":"float","description":"Focal length at depth resolution."},
        {"name":"labels","type":"list[int] | None","description":"Optional object labels."},
        {"name":"n_frames, stride, tol, grow, keep_out","type":"numbers","description":"Original plane fitter parameters."}],
    "outputs":[{"name":"plane","type":"dict | None","description":"Metric plane with corrected camera pitch/roll and original angle diagnostics."}],
    "entrypoint":"fit_camera_to_table_plane_cv", "requirements":["numpy","scipy","pillow"],
    "tags":["camera","depth","plane","calibration"]
}

def fit_camera_to_table_plane_cv(depth_npz,masks_dir,focal_px,labels=None,n_frames=5,stride=3,tol=.008,grow=30,keep_out=4):
    plane=fit_camera_to_table_plane(depth_npz,masks_dir,focal_px,labels=labels,n_frames=n_frames,
        stride=stride,tol=tol,grow=grow,keep_out=keep_out)
    if plane is None:return None
    up=plane['up']
    plane['original_angle_diagnostics']={k:plane[k] for k in ('pitch_deg','roll_deg')}
    plane['pitch_deg']=round(math.degrees(math.asin(max(-1,min(1,-up[2])))),3)
    plane['roll_deg']=round(math.degrees(math.atan2(up[0],-up[1])),3)
    plane['angle_convention']='common.rigcam.R_cv_from_angles: positive pitch below horizon'
    return plane

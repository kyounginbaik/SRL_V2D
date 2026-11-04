"""Source-derived measurement diagnostics; no Stage 2 acceptance claim."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps/DexAgent-adapted'))
from tool_library.skills.segment_source_frame import segment_source_frame
from tool_library.skills.fit_camera_to_table_plane_cv import fit_camera_to_table_plane_cv
from tool_library.skills.register_table_camera import register_table_camera
from tool_library.skills.measure_mask_footprint import measure_mask_footprint

out=ROOT/'continuation/chopsticks'
video=ROOT/'clips/02_chopsticks_grape.mp4'
plate=segment_source_frame(str(video),0,[224,190,325,294],str(out/'plate_mask'))
try:
    grape=segment_source_frame(str(video),0,[337,180,365,212],str(out/'grape_mask'))
except ValueError as error:
    grape=None;grape_error=str(error)
(out/'depth_masks').mkdir(exist_ok=True)
image=cv2.imread(plate['mask'],cv2.IMREAD_GRAYSCALE)
cv2.imwrite(str(out/'depth_masks/frame00000.png'),(image>0).astype(np.uint8))
plane=fit_camera_to_table_plane_cv(str(out/'depth.npz'),str(out/'depth_masks'),100.,n_frames=1,grow=35)
(out/'table_plane.json').write_text(json.dumps(plane,indent=2))
camera=register_table_camera(str(out/'camera.json'),source='estimate',width=512,height=512,focal_px=200.,
    pitch_deg=plane['pitch_deg'],roll_deg=plane['roll_deg'],height_m=plane['camera_height_m'],frame=plate['source_frame'])
measurements={'plate':measure_mask_footprint(plate['mask'],camera['camera'])}
if grape is not None:measurements['grape']=measure_mask_footprint(grape['mask'],camera['camera'])
else:measurements['grape']={'segmentation_error':grape_error}
report={'scope':'Measurements only. CPU masks were computed from the source but not certified by a Stage 2 harness.',
    'source_clip_frame':0,'recorded_depth_source_frame':180,'source_recording_start_s':9.,
    'plane':plane,'measurements':measurements,'missing_native_goal':'Grape held through opposed contacts at two separate chopstick tips; no direct hand-grape grasp.'}
(out/'measurements.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)

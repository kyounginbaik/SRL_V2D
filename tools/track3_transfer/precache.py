import json,sys,time
from pathlib import Path
import newton,numpy as np
from flash_chord.assets.mesh import convex_decompose
b=Path(__file__).resolve().parent.parent
path=json.loads((b/'collision_manifest.json').read_text())[int(sys.argv[1])]
print('Precomputing collision',path,flush=True)
builder=newton.ModelBuilder();builder.add_urdf(path,floating=True,enable_self_collisions=False,collapse_fixed_joints=True,ignore_inertial_definitions=False)
for i in range(builder.shape_count):
 if builder.shape_type[i]==newton.GeoType.MESH and int(builder.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
  mesh=builder.shape_source[i];start=time.time();hulls=convex_decompose(np.asarray(mesh.vertices),np.asarray(mesh.indices).reshape(-1,3));print('Hulls',len(hulls),'seconds',time.time()-start,flush=True)

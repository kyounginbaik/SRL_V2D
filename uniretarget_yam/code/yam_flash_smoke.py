"""CPU check of the YAM flash_chord embodiment: scene builds, reference binds, physics steps at zero residual, spoon tracked."""
import json,sys
from pathlib import Path
import numpy as np
import warp as wp
wp.config.verbose=False
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
import flash_chord
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.setup import *
import flash_chord.scene.setup as S
parquet,urdf=sys.argv[1:3]
root=Path(flash_chord.__file__).parent/'configs'
with initialize_config_dir(str(root),version_base='1.3'):
    cfg=compose('train_flash_sac',overrides=['experiment=yam_sharpa_flash_sac',f"task.parquet='{parquet}'",f"embodiment.config.urdf_path._args_=['{urdf}']",'scene.world_count=2'])
print('composed',cfg.embodiment._target_,cfg.task.motion_speed,cfg.training.total_environment_steps,flush=True)
emb=instantiate_typed(cfg.embodiment,Embodiment);print('embodiment',type(emb).__name__,flush=True)
fn=[n for n in dir(S) if 'setup' in n.lower() or 'build' in n.lower()];print(fn)

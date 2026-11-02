import sys,json
from flash_chord.training.checkpoint import read_checkpoint_metadata as r
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY as K
c=json.loads(r(sys.argv[1])[K]);print(list(c.keys()));print('task',json.dumps(c['task'])[:500]);print('scene',json.dumps(c['scene'])[:400]);print('embodiment',json.dumps(c['embodiment'])[:200])

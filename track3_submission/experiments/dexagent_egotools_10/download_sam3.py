from pathlib import Path
import json
from huggingface_hub import hf_hub_download, get_token
root=Path(__file__).resolve().parent
path=hf_hub_download("facebook/sam3", "sam3.pt", revision="3c879f39826c281e95690f02c7821c4de09afae7", cache_dir=root/"models/hf/hub", token=get_token())
print(json.dumps({"checkpoint":path}),flush=True)

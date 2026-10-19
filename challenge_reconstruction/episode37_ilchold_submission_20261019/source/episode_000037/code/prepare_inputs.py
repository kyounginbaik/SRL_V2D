"""Freeze approved references and assets; only rebase asset paths for runtime."""
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    manifest = []
    for item in json.loads((BASE / "inputs.json").read_text()):
        episode = item["episode"]
        source = Path(item["reference"])
        if digest(source) != item["sha256"]:
            raise ValueError(f"Approved reference changed: {source}")
        dest = BASE / "frozen" / episode
        dest.mkdir(parents=True, exist_ok=True)
        frozen = dest / "approved_reference.parquet"
        if frozen.exists() and digest(frozen) != item["sha256"]:
            raise ValueError(f"Frozen reference changed: {frozen}")
        if not frozen.exists():
            shutil.copy2(source, frozen)
        assets = dest / "assets"
        if assets.exists():
            for original in source.parent.joinpath("assets").rglob("*"):
                if original.is_file():
                    copy = assets / original.relative_to(source.parent / "assets")
                    if not copy.is_file() or digest(copy) != digest(original):
                        raise ValueError(f"Frozen asset differs: {copy}")
        else:
            shutil.copytree(source.parent / "assets", assets)
        table = pq.read_table(frozen)
        row = table.to_pylist()[0]
        if row["safe_object_body_names"] != item["objects"]:
            raise ValueError(f"Object roster mismatch: {episode}")
        for key in ("object_mesh_paths", "object_urdf_paths"):
            new_paths = []
            for old in row[key]:
                old = Path(old)
                if old.parent != source.parent / "assets":
                    raise ValueError(f"Unexpected asset path: {old}")
                new_paths.append(str(assets / old.name))
            row[key] = new_paths
        runtime = dest / "reference.parquet"
        if not runtime.exists():
            arrays = [pa.array([row[name]], type=table.schema.field(name).type) for name in table.column_names]
            pq.write_table(pa.Table.from_arrays(arrays, schema=table.schema), runtime)
        actual = pq.read_table(runtime).to_pylist()[0]
        for key in row:
            if actual[key] != row[key]:
                raise ValueError(f"Runtime reference mismatch in {key}: {episode}")
        if len(row["object_body_position"]) != item["frames"]:
            raise ValueError(f"Frame count mismatch: {episode}")
        asset_hashes = {str(p.relative_to(dest)): digest(p) for p in sorted(assets.rglob("*")) if p.is_file()}
        manifest.append({"episode": episode, "approved_sha256": item["sha256"],
                         "runtime_sha256": digest(runtime), "asset_sha256": asset_hashes,
                         "frames": item["frames"], "fps": item["fps"], "objects": item["objects"]})
    (BASE / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()

#!/usr/bin/env python3
"""Freeze a portable shared training campaign for selected public Track 3 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RECIPE = Path(__file__).resolve().parent
SOURCE = RECIPE.parents[1]
sys.path.insert(0, str(RECIPE / "submission"))
from contract import atomic_json, digest, load_roster, validate_kit_roster
sys.path.insert(0, str(RECIPE / "cluster"))
from initial_state import validate_initial_state_report

IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".cache", "*.pyc")


def hashes_under(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*"))
            if p.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".cache")
                                       for part in p.relative_to(path).parts)}


def reference_controller_limits(row):
    """Measure unchanged reference motion and bound physical speeds conservatively."""
    fps = float(row["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Reference FPS must be finite and positive")
    hands, tracks = {}, {}
    for side in ("left", "right"):
        position = np.asarray(row[f"robot_{side}_wrist_position"], dtype=np.float64)
        quaternion = np.asarray(row[f"robot_{side}_wrist_wxyz"], dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3 or quaternion.shape != (len(position), 4):
            raise ValueError(f"Invalid {side} wrist reference dimensions")
        if len(position) < 2 or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError(f"Invalid {side} wrist reference values")
        norms = np.linalg.norm(quaternion, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError(f"Zero {side} wrist reference quaternion")
        # Normalize this temporary measurement array only. Reference arrays are
        # never filtered, scaled, clipped, or rewritten by the limiter setup.
        quaternion = quaternion / norms[:, None]
        linear = np.linalg.norm(np.diff(position, axis=0), axis=1) * fps
        cosine = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=1))
        angular = 2 * np.arccos(np.clip(cosine, 0., 1.)) * fps
        tracks[side] = (linear, angular)
        segment = (np.arange(len(linear)) / fps >= 6.) & (np.arange(len(linear)) / fps < 10.)
        hands[side] = dict(linear_speed_max_m_s=float(linear.max()),
                           angular_speed_max_rad_s=float(angular.max()),
                           segment_6_to_10_seconds=dict(
                               transitions=int(segment.sum()),
                               linear_speed_max_m_s=float(linear[segment].max()) if segment.any() else None,
                               angular_speed_max_rad_s=float(angular[segment].max()) if segment.any() else None))
    linear = max(item["linear_speed_max_m_s"] for item in hands.values())
    angular = max(item["angular_speed_max_rad_s"] for item in hands.values())
    linear_limit, angular_limit = min(2., max(1., 1.2 * linear)), min(20., max(12., 1.2 * angular))
    for side, (linear_track, angular_track) in tracks.items():
        hands[side].update(
            linear_reference_exceedance_count=int(np.count_nonzero(linear_track > linear_limit)),
            angular_reference_exceedance_count=int(np.count_nonzero(angular_track > angular_limit)))
    covered = linear <= linear_limit and angular <= angular_limit
    return dict(linear_speed_limit_m_s=linear_limit,
                angular_speed_limit_rad_s=angular_limit,
                reference_linear_speed_max_m_s=linear,
                reference_angular_speed_max_rad_s=angular, margin=1.2,
                linear_minimum_m_s=1., angular_minimum_rad_s=12.,
                linear_hard_ceiling_m_s=2., angular_hard_ceiling_rad_s=20.,
                reference_peak_covered=bool(covered),
                reconstruction_speed_quality_flag=("reference_speed_exceeds_physical_ceiling" if not covered else None),
                reference_motion_modified=False, hands=hands,
                rule="min(hard ceiling, max(minimum, 1.2 * reference peak)); reference unchanged; residual limiting is separate")


def _source_path(value, reference):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (reference.parent / path).resolve()


def _freeze_assets(source_reference, row, destination):
    """Copy the episode closure and remove producer paths from runtime assets."""
    source_assets = source_reference.parent / "assets"
    if not source_assets.is_dir():
        raise ValueError(f"No sibling asset closure: {source_reference}")
    assets = destination / "assets"
    shutil.copytree(source_assets, assets, ignore=IGNORE)
    paths = {}
    for field in ("object_mesh_paths", "object_urdf_paths"):
        paths[field] = []
        for value in row[field]:
            original = _source_path(value, source_reference)
            # Approved references can retain a previous producer prefix while
            # their frozen sibling assets are the intended portable closure.
            # Prefer that declared local copy, even if the producer still exists.
            marker = str(value).rfind("/assets/")
            if marker >= 0:
                sibling = source_assets / str(value)[marker + len("/assets/"):]
                if sibling.is_file():
                    original = sibling.resolve()
            elif str(value).startswith("assets/"):
                original = (source_reference.parent / value).resolve()
            if not original.is_file():
                raise FileNotFoundError(original)
            try:
                relative = original.relative_to(source_assets.resolve())
            except ValueError as error:
                raise ValueError(f"Asset is outside the episode closure: {original}") from error
            target = assets / relative
            if not target.is_file():
                raise FileNotFoundError(target)
            paths[field].append(str(target))
    # Make every URDF mesh path local. Preserve all physical parameters.
    for original in source_assets.rglob("*.urdf"):
        relative = original.relative_to(source_assets)
        target = assets / relative
        tree = ET.parse(original)
        changed = False
        for mesh in tree.findall(".//mesh"):
            value = mesh.attrib["filename"]
            if "://" in value:
                raise ValueError(f"Unsupported external URDF mesh URI: {original}: {value}")
            source_mesh = Path(value)
            source_mesh = source_mesh.resolve() if source_mesh.is_absolute() else (original.parent / source_mesh).resolve()
            try:
                copied_mesh = assets / source_mesh.relative_to(source_assets.resolve())
            except ValueError as error:
                raise ValueError(f"URDF mesh escapes episode closure: {source_mesh}") from error
            if not copied_mesh.is_file():
                raise FileNotFoundError(copied_mesh)
            local = os.path.relpath(copied_mesh, target.parent)
            if value != local:
                mesh.set("filename", local)
                changed = True
        if changed:
            tree.write(target, encoding="utf-8", xml_declaration=True)
    # The provided closures are untextured OBJ/PLY meshes. Explicitly reject an
    # unresolved material dependency rather than silently copying partial assets.
    for obj in assets.rglob("*.obj"):
        with obj.open(errors="strict") as stream:
            for line in stream:
                if line.startswith("mtllib "):
                    for material in line.split()[1:]:
                        candidate = (obj.parent / material).resolve()
                        if assets.resolve() not in candidate.parents or not candidate.is_file():
                            raise ValueError(f"Unresolved OBJ material dependency: {obj}: {material}")
    return paths


def freeze_episode(item, root, roster, support_root=None):
    episode = item["episode"]
    index = str(int(episode.removeprefix("episode_")))
    source = Path(item["source_reference"]).resolve()
    source_hash = digest(source)
    if source_hash != item["reference_sha256"]:
        raise ValueError(f"Reference changed after inventory: {source}")
    table = pq.ParquetFile(source).read()
    if table.num_rows != 1:
        raise ValueError(f"Expected exactly one reference row: {source}")
    row = table.to_pylist()[0]
    expected = roster[index]
    frames, fps = len(row["object_body_position"]), float(row["fps"])
    if row["object_body_names"] != expected["object_names"]:
        raise ValueError(f"Wrong public object order: {episode}")
    if expected["frame_indices"] != list(range(frames)) or expected["world_indices"] != [0]:
        raise ValueError(f"Reference does not cover the exact public frame roster: {episode}")
    if frames != item["frames"] or fps != float(item["fps"]):
        raise ValueError(f"Reference timing changed after inventory: {episode}")
    limits = reference_controller_limits(row)
    destination = root / "frozen" / episode
    destination.mkdir(parents=True)
    shutil.copy2(source, destination / "source_reference.parquet")
    changed = _freeze_assets(source, row, destination)
    support_asset = None
    ground_plane = None
    initialization = {}
    if support_root is not None:
        support = Path(support_root) / episode
        if (support / 'support.usda').is_file():
            for name in ('support.usda', 'support_report.json'):
                shutil.copy2(support / name, destination / 'assets' / name)
            shutil.copytree(support, destination / 'assets' / 'support_validation', ignore=IGNORE)
            support_asset = 'assets/support.usda'
            support_report = json.loads((support / 'support_report.json').read_text())
            if not support_report.get('physics_passed'):
                raise ValueError(f'Measured support has not passed initialization physics: {episode}')
            if support_report.get('source_reference_sha256') != source_hash:
                raise ValueError(f'Measured support belongs to a different reference: {episode}')
            if support_report.get('support_sha256') != digest(support / 'support.usda'):
                raise ValueError(f'Measured support asset changed after validation: {episode}')
            floor = support_report.get('measured_floor')
            if floor:
                ground_plane = [*floor['normal'], floor['offset']]
            initial_report = support / 'initial_state_report.json'
            if initial_report.is_file():
                initialization = validate_initial_state_report(initial_report,
                    episode=episode, source_sha256=source_hash, frozen_directory=destination,
                    object_names=row['object_body_names'], object_urdf_paths=changed['object_urdf_paths'])
                shutil.copy2(initial_report, destination/'assets/initial_state_report.json')
                initialization['initial_state_report'] = 'assets/initial_state_report.json'
    for field, values in changed.items():
        field_index = table.schema.get_field_index(field)
        table = table.set_column(field_index, table.schema.field(field),
                                 pa.array([values], type=table.schema.field(field).type))
    runtime = destination / "reference.parquet"
    pq.write_table(table, runtime)
    # Asset-path rewriting must preserve every numerical/reference field exactly,
    # including the fast 6--10 second motion in episode 1.
    original_table = pq.ParquetFile(source).read()
    unchanged = [name for name in original_table.column_names if name not in changed]
    if not table.select(unchanged).equals(original_table.select(unchanged)):
        raise ValueError(f"Runtime conversion changed reference motion: {episode}")
    masses = {}
    for name, path in zip(row["object_body_names"], changed["object_urdf_paths"], strict=True):
        masses[name] = sum(float(m.attrib["value"]) for m in ET.parse(path).findall(".//inertial/mass"))
    record = dict(episode=episode, episode_index=int(index), frames=frames, fps=fps,
                  objects=row["object_body_names"], source_reference=str(source),
                  source_sha256=source_hash, approved_sha256=source_hash,
                  runtime_sha256=digest(runtime),
                  asset_sha256={f"assets/{key}": value for key, value in hashes_under(destination / "assets").items()},
                  object_masses_kg=masses, controller_limits=limits, support_asset=support_asset,
                  ground_plane=ground_plane,
                  reference_description=item["quality_note"], source_kind=item["source_kind"],
                  source_quality=item["quality"], reference_values_preserved=True,
                  asset_path_policy="prepared-root paths; loader resolves sibling assets after relocation")
    record.update(initialization)
    if "white_pot" in masses:
        record["pot_mass_kg"] = masses["white_pot"]
    if "white_pot_lid" in masses:
        record["lid_mass_kg"] = masses["white_pot_lid"]
    atomic_json(destination / "preparation.json", record)
    return record


def copy_public_kit(source, destination):
    """Copy only participant-facing packer code, config and public Track 3 grids."""
    validate_kit_roster(source)
    destination.mkdir()
    shutil.copytree(source / "v2dlb", destination / "v2dlb", ignore=IGNORE)
    for relative in ("tools/pack_submission.py", "tools/pack_reconstruction.py", "submit/v2d_submit.py", "config/tracks.json",
                     "data/track_3_evaluation_objects.json", "data/track_3_sample_submission.csv",
                     "data/track_3_mesh_sample_submission.parquet", "requirements.txt"):
        original = source / relative
        if not original.is_file():
            raise FileNotFoundError(original)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    atomic_json(destination / "copy_manifest.json", dict(
        purpose="Public participant packer subset; no hidden reference poses or labels",
        source=str(source), files=hashes_under(destination)))


def prepare(inventory_path, root, *, priors_root, kit, seed=43, upper_edit_scope='full', support_root=None):
    inventory_path, root = Path(inventory_path).resolve(), Path(root).resolve()
    priors_root, kit = Path(priors_root).resolve(), Path(kit).resolve()
    if root.exists():
        raise ValueError(f"Fresh campaign only; destination already exists: {root}")
    if SOURCE == root or SOURCE in root.parents:
        raise ValueError("Choose a run root outside the FlashCHORD source tree")
    inventory = json.loads(inventory_path.read_text())
    roster = load_roster()
    items = inventory["episodes"]
    ids = [str(item["episode_index"]) for item in items]
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(roster):
        raise ValueError("Input inventory must select a nonempty, unique subset of public episodes")
    validate_kit_roster(kit)
    prior_hashes = {str(Path(name).relative_to("priors")): expected
                    for name, expected in json.loads((RECIPE / "data_manifest.json").read_text()).items()
                    if name.startswith("priors/")}
    for name, expected in prior_hashes.items():
        if digest(priors_root / name) != expected:
            raise ValueError(f"PCA prior differs from the verified recipe: {priors_root / name}")
    for path in (SOURCE / "src/flash_chord/assets").rglob("*"):
        if path.is_file() and path.stat().st_size < 1024:
            if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Unmaterialized Git LFS asset: {path}")
    root.mkdir(parents=True)
    for directory in ("reports", "warp_cache", "slurm", "submission_results/smoke", "submission_results/pilot"):
        (root / directory).mkdir(parents=True)
    shutil.copytree(SOURCE, root / "flash_chord", ignore=shutil.ignore_patterns(
        "experiments", "__pycache__", ".pytest_cache", ".cache", ".venv", "*.pyc",
        "outputs", "logs", "wandb", "*.egg-info", ".git"))
    shutil.copytree(RECIPE / "cluster", root / "cluster", ignore=IGNORE)
    shutil.copytree(RECIPE / "submission", root / "submission", ignore=IGNORE)
    shutil.copy2(RECIPE / "run.sh", root / "run.sh")
    if (RECIPE / "submit_batch.py").is_file():
        shutil.copy2(RECIPE / "submit_batch.py", root / "submit_batch.py")
    (root / "priors").mkdir()
    for name in prior_hashes:
        destination = root / "priors" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(priors_root / name, destination)
    copy_public_kit(kit, root / "submission_kit")
    records = {}
    for item in sorted(items, key=lambda value: value["episode_index"]):
        episode = item["episode"]
        record = freeze_episode(item, root, roster, support_root=support_root)
        records[episode] = record
        base = root / episode
        base.mkdir()
        # Keep real copies: these scripts derive the episode from __file__.resolve().
        shutil.copytree(RECIPE / "code", base / "code", ignore=IGNORE)
        atomic_json(base / "frozen_manifest.json", [record])
        for name in ("locks", "slurm", "runs"):
            (base / name).mkdir()
        for name in ("flash_chord", "frozen", "priors", "warp_cache"):
            (base / name).symlink_to("../" + name, target_is_directory=True)
        print(f"Prepared {episode}: {record['frames']} frames, "
              f"wrist caps {record['controller_limits']['linear_speed_limit_m_s']:.3g} m/s, "
              f"{record['controller_limits']['angular_speed_limit_rad_s']:.3g} rad/s", flush=True)
    shutil.copy2(inventory_path, root / "input_inventory.json")
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    batch = dict(episodes=list(records), episode_count=len(records), seed=int(seed), seeds=[int(seed)],
                 upper_edit_scope=upper_edit_scope,
                 records=records, phases=["smoke", "pilot"],
                 public_episode_count=len(roster),
                 missing_public_episodes=[f'episode_{int(key):06d}' for key in sorted(roster, key=int)
                                          if f'episode_{int(key):06d}' not in records],
                 public_coverage_complete=len(records) == len(roster),
                 main_training_transitions_per_episode=100663296,
                 first_pass_transitions_per_episode=1048576,
                 public_roster="submission/public_roster.json",
                 public_roster_sha256=digest(root / "submission/public_roster.json"),
                 submission_kit="submission_kit", submission_results="submission_results",
                 source_git_revision=revision, input_inventory_sha256=digest(inventory_path),
                 input_quality_summary=dict(
                     approved_prepared_episodes=sum(r['source_kind'] == 'approved_finite_response_frozen' for r in records.values()),
                     latest_final_recon_episodes=sum(r['source_kind'] == 'final_recon' for r in records.values()),
                     older_runnable_episodes=sum(r['source_kind'] not in ('approved_finite_response_frozen', 'final_recon') for r in records.values()),
                     reference_speed_exceedance_episodes=[name for name, r in records.items()
                                                         if not r['controller_limits']['reference_peak_covered']]),
                 reference_motion_modified=False,
                 selection="archived best unassisted prefix then physical cost; not hidden challenge score",
                 resume_supported=False)
    atomic_json(root / "batch_manifest.json", batch)
    atomic_json(root / "source_manifest.json", hashes_under(root / "flash_chord"))
    launcher_hashes = {}
    for directory in [root / "cluster", root / "submission", root / "submission_kit",
                      *(root / episode / "code" for episode in records)]:
        launcher_hashes.update({str(directory.relative_to(root) / name): value
                                for name, value in hashes_under(directory).items()})
    launcher_hashes["run.sh"] = digest(root / "run.sh")
    if (root / "submit_batch.py").is_file():
        launcher_hashes["submit_batch.py"] = digest(root / "submit_batch.py")
    atomic_json(root / "launcher_manifest.json", launcher_hashes)
    data_hashes = {}
    for directory in (root / "frozen", root / "priors"):
        data_hashes.update({str(directory.relative_to(root) / name): value
                            for name, value in hashes_under(directory).items()})
    for relative in ["input_inventory.json", "batch_manifest.json",
                     *(f"{episode}/frozen_manifest.json" for episode in records)]:
        data_hashes[relative] = digest(root / relative)
    atomic_json(root / "data_manifest.json", data_hashes)
    atomic_json(root / "reports/preparation.json", dict(
        source=str(SOURCE), run_root=str(root), git_revision=revision,
        preparation_script_sha256=digest(Path(__file__)), episode_count=len(records),
        source_manifest_sha256=digest(root / "source_manifest.json"),
        launcher_manifest_sha256=digest(root / "launcher_manifest.json"),
        data_manifest_sha256=digest(root / "data_manifest.json"),
        batch_manifest_sha256=digest(root / "batch_manifest.json")))
    print(f"Prepared {len(records)} episodes at {root}\n"
          f"CPU preflight: bash {root}/run.sh preflight", flush=True)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--priors-root", required=True, type=Path)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument('--upper-edit-scope', choices=('full', 'wrists_only'), default='full')
    parser.add_argument('--support-root', type=Path,
                        help='Measured per-episode support.usda and support_report.json directories')
    args = parser.parse_args()
    prepare(args.inventory, args.run_root, priors_root=args.priors_root, kit=args.kit, seed=args.seed,
            upper_edit_scope=args.upper_edit_scope, support_root=args.support_root)


if __name__ == "__main__":
    main()

"""
Example usage:
python scripts/retain_merge.py --args.base-checkpoint gs://openpi-assets/checkpoints/pi0_base/params --args.finetuned-checkpoint /root/.cache/huggingface/hub/models--F-Fer-
-tasks-merged/snapshots/c96dae45d8a991b936b5fa6d67366759e7da8a09/59999/params/ --args.output-dir ./checkpoints/pi0_ur_tasks_merged_retain_0.5/params --args.alpha 0.5 --args
"""

import dataclasses
import logging
import pathlib
import shutil

import numpy as np
import orbax.checkpoint as ocp
import tyro

from flax import traverse_util
from openpi.models import model
from openpi.shared import download

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("retain_merge")


@dataclasses.dataclass
class Args:
    """Arguments for RETAIN weight merging script."""

    # Path or URL to the base model checkpoint.
    # Example: "gs://openpi-assets/checkpoints/pi0_base/params"
    base_checkpoint: str

    # Path or URL to the finetuned model checkpoint.
    # Example: "./checkpoints/pi0_ur_tasks_merged/<exp>/<step>/params"
    finetuned_checkpoint: str

    # Directory where the merged checkpoint will be saved.
    output_dir: pathlib.Path

    # Merging coefficient alpha.
    # theta_new = (1 - alpha) * theta_base + alpha * theta_ft
    # alpha = 0.0 -> Base model
    # alpha = 1.0 -> Finetuned model
    alpha: float = 0.5

    # Whether to overwrite the output directory if it exists.
    overwrite: bool = False


def load_params(path: str, name: str) -> dict:
    """Load parameters from a checkpoint path using CPU memory (numpy)."""
    logger.info(f"Downloading/Resolving {name} checkpoint: {path}")
    local_path = download.maybe_download(path)

    logger.info(f"Loading {name} parameters from {local_path}...")
    # restore_type=np.ndarray forces loading into System RAM, avoiding GPU VRAM.
    return model.restore_params(local_path, restore_type=np.ndarray)


def merge_trees(base_tree, ft_tree, alpha: float):
    """Linearly interpolate between two parameter trees with component-specific logic.

    Merging Logic:
    1. PaliGemma (Vision + LLM Backbone): Interpolate using alpha
       - base keys often have no suffix or simple names.
    2. Action Expert: KEEP from Finetuned (alpha=1.0)
       - Identified by `_1` suffix in the parameter names (e.g. `attn_1`, `mlp_1`) inside PaliGemma structure
         or specific top-level keys if any.
    3. Projectors (e.g. `action_in_proj`, `action_out_proj`): KEEP from Finetuned (alpha=1.0)
       - These are task-specific (dimension dependent).
    """

    # Flatten dictionaries to work with keys easily
    flat_base = traverse_util.flatten_dict(base_tree, sep="/")
    flat_ft = traverse_util.flatten_dict(ft_tree, sep="/")

    merged_flat = {}

    all_keys = set(flat_base.keys()) | set(flat_ft.keys())

    for key in all_keys:
        if key not in flat_base:
            # If only in FT, must be new task-specific param -> Keep FT
            merged_flat[key] = flat_ft[key]
            continue
        if key not in flat_ft:
            # If only in Base, rare but keep Base
            merged_flat[key] = flat_base[key]
            continue

        base_val = flat_base[key]
        ft_val = flat_ft[key]

        # Ensure shapes match
        if base_val.shape != ft_val.shape:
            # If shapes mismatch (e.g. different vocab or action dim), prefer FT
            logger.warning(f"Shape mismatch for {key}: {base_val.shape} vs {ft_val.shape}. Keeping Finetuned.")
            merged_flat[key] = ft_val
            continue

        # Determine merging strategy based on key name
        # Action Expert parameters in Pi0 typically have `_1` in their layer names inside PaliGemma
        # or are part of the projectors.
        # Structure in Pi0:
        # - PaliGemma/llm/... (Backbone)
        # - PaliGemma/img/... (Vision)
        # - action_in_proj, action_out_proj (Projectors)
        # - state_proj, action_time_mlp_in, action_time_mlp_out (Projectors/Adapters)

        is_action_expert = "_1" in key  # Heuristic for Action Expert layers in shared backbone
        is_projector = any(
            k in key
            for k in [
                "action_in_proj",
                "action_out_proj",
                "state_proj",
                "time_mlp_in",
                "time_mlp_out",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ]
        )

        if is_action_expert or is_projector:
            # Keep Finetuned (alpha = 1.0)
            merged_flat[key] = ft_val
        else:
            # Vision + LLM Backbone -> Interpolate
            merged_flat[key] = (1.0 - alpha) * base_val + alpha * ft_val

    # Unflatten back to nested structure
    return traverse_util.unflatten_dict(merged_flat, sep="/")


def main(args: Args):
    if args.output_dir.exists():
        if args.overwrite:
            logger.info(f"Output directory {args.output_dir} exists. Overwriting...")
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(f"Output directory {args.output_dir} already exists. Use --overwrite to replace it.")

    # Load checkpoints
    base_params = load_params(args.base_checkpoint, "Base")
    ft_params = load_params(args.finetuned_checkpoint, "Finetuned")

    # Merge
    merged_params = merge_trees(base_params, ft_params, args.alpha)

    # Save
    logger.info(f"Saving merged checkpoint to {args.output_dir}...")

    # Create the directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(args.output_dir.resolve(), {"params": merged_params}, force=args.overwrite)

    # Attempt to copy assets (e.g. norm_stats) from the finetuned checkpoint
    ft_local_path = download.maybe_download(args.finetuned_checkpoint)
    src_assets = ft_local_path.parent / "assets" if ft_local_path.name == "params" else ft_local_path / "assets"

    if src_assets.exists():
        dest_assets = (
            args.output_dir.parent / "assets" if args.output_dir.name == "params" else args.output_dir / "assets"
        )

        should_copy = True
        if dest_assets.exists():
            if args.overwrite:
                logger.info(f"Overwriting assets at {dest_assets}...")
                shutil.rmtree(dest_assets)
            else:
                logger.warning(f"Assets directory {dest_assets} already exists. Skipping copy.")
                should_copy = False

        if should_copy:
            logger.info(f"Copying assets from {src_assets} to {dest_assets}...")
            # symlinks=False ensures we copy the actual files, not links
            shutil.copytree(src_assets, dest_assets, symlinks=False)
    else:
        logger.info(f"No assets found at {src_assets}. Skipping assets copy.")

    logger.info("Done!")
    logger.info(f"Merged model saved to: {args.output_dir}")
    logger.info("To use this checkpoint in training/inference, set:")
    logger.info(f'weight_loader=weight_loaders.CheckpointWeightLoader("{args.output_dir}")')


if __name__ == "__main__":
    tyro.cli(main)

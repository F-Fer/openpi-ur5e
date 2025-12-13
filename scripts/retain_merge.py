
import dataclasses
import logging
import pathlib

import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro

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
    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi0_base/params"

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
    params = model.restore_params(local_path, restore_type=np.ndarray)
    return params


def merge_trees(base_tree, ft_tree, alpha: float):
    """Linearly interpolate between two parameter trees."""
    
    def _merge_leaf(base_leaf, ft_leaf):
        # Ensure we are working with matching shapes/types
        if base_leaf.shape != ft_leaf.shape:
            raise ValueError(f"Shape mismatch: {base_leaf.shape} vs {ft_leaf.shape}")
        
        # Perform interpolation: (1 - alpha) * base + alpha * ft
        # Doing this in numpy keeps it on CPU.
        return (1.0 - alpha) * base_leaf + alpha * ft_leaf

    logger.info(f"Merging parameters with alpha={alpha}...")
    # jax.tree.map works with arbitrary PyTrees (dicts, lists, etc.) and numpy arrays
    merged_tree = jax.tree.map(_merge_leaf, base_tree, ft_tree)
    return merged_tree


def main(args: Args):
    if args.output_dir.exists():
        if args.overwrite:
            logger.info(f"Output directory {args.output_dir} exists. Overwriting...")
        else:
            raise FileExistsError(
                f"Output directory {args.output_dir} already exists. Use --overwrite to replace it."
            )

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
    checkpointer.save(
        args.output_dir.resolve(), 
        {"params": merged_params}
    )
    
    logger.info("Done!")
    logger.info(f"Merged model saved to: {args.output_dir}")
    logger.info("To use this checkpoint in training/inference, set:")
    logger.info(f'weight_loader=weight_loaders.CheckpointWeightLoader("{args.output_dir}")')


if __name__ == "__main__":
    tyro.cli(main)


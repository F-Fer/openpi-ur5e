import numpy as np
import einops
import dataclasses
import logging

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger("openpi")
logger.setLevel(logging.DEBUG)

def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5E policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class UR5EInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["observation/joint_position"])

        if "observation/gripper_position" in data:
            gripper_pos = np.asarray(data["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([joints, gripper_pos])
        else:
            # Gripper position is embedded in the state.
            if joints.ndim.shape[-1] == 7:
                state = joints
            if joints.ndim.shape[-1] == 6:
                # No gripper position in the state. Add a zero.
                np.concatenate([joints, np.zeros(1, dtype=joints.dtype)])
            else:
                raise ValueError(f"Joints dimension is {joints.ndim}, expected 6 or 7.")

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # We don't mask out padding images for FAST models.
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        logger.debug(f"Inputs: {inputs}")

        return inputs

@dataclasses.dataclass(frozen=True)
class UR5EOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
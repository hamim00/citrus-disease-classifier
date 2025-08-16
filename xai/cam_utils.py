# xai/cam_utils.py
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _tensor_to_rgb01(t: torch.Tensor) -> np.ndarray:
    # t: 1x3xHxW, ImageNet-normalized -> HxWx3 in [0,1]
    arr = t[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0.0, 1.0)

def _squeeze_cam(cam_arr) -> np.ndarray:
    # Ensure HxW float32 in [0,1]; handle (N,H,W) or list outputs
    while isinstance(cam_arr, (list, tuple)):
        cam_arr = cam_arr[0]
    cam_arr = np.array(cam_arr)
    while cam_arr.ndim > 2:
        cam_arr = cam_arr[0]
    cam_arr = cam_arr.astype(np.float32)
    mn, mx = cam_arr.min(), cam_arr.max()
    if mx > mn:
        cam_arr = (cam_arr - mn) / (mx - mn)
    else:
        cam_arr = np.zeros_like(cam_arr, dtype=np.float32)
    return cam_arr

_CAM_MAP = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "ablationcam": AblationCAM,
}

def run_cam(
    method_name: str,
    model: torch.nn.Module,
    target_layer,
    input_tensor: torch.Tensor,          # 1x3xHxW, normalized; on same device as model
    target_category: int | None = None,
    eigen_smooth: bool = False,
    aug_smooth: bool = False,
) -> Image.Image:
    name = method_name.lower()
    if name not in _CAM_MAP:
        raise ValueError(f"Unknown CAM method: {method_name}")
    cam_cls = _CAM_MAP[name]

    # New API: no 'use_cuda' — device inferred from input_tensor/model
    with cam_cls(model=model, target_layers=[target_layer]) as cam:
        targets = None
        if target_category is not None:
            targets = [ClassifierOutputTarget(int(target_category))]
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth,
        )

    grayscale_cam = _squeeze_cam(grayscale_cam)
    rgb = _tensor_to_rgb01(input_tensor)
    vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # returns uint8 RGB
    return Image.fromarray(vis)

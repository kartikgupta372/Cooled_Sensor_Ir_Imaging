# pipeline/enhance.py
# Re-exports everything from src/enhance.py
# This allows: from pipeline.enhance import enhance_frame, detect_image_type, ...

from enhance import (
    detect_image_type,
    robust_normalize,
    adaptive_normalize,
    inpaint_artifacts,
    anisotropic_diffusion,
    multi_scale_retinex,
    tophat_enhance,
    apply_clahe,
    bilateral_denoise,
    unsharp_mask,
    ir_gamma,
    super_resolve,
    tone_map,
    enhance_frame,
)

__all__ = [
    "detect_image_type",
    "robust_normalize",
    "adaptive_normalize",
    "inpaint_artifacts",
    "anisotropic_diffusion",
    "multi_scale_retinex",
    "tophat_enhance",
    "apply_clahe",
    "bilateral_denoise",
    "unsharp_mask",
    "ir_gamma",
    "super_resolve",
    "tone_map",
    "enhance_frame",
]

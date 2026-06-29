"""
pipeline/enhance.py — Enhancement proxy for the Streamlit app (v3.0)
=====================================================================
This is the file imported by app.py via:
    from pipeline.enhance import enhance_frame, detect_image_type

It delegates to the core src/enhance.py v3.0 pipeline which contains
all the improved algorithms (Anisotropic Diffusion + MSR + Bilateral).

New algorithm chain (delegated to src/enhance.py):
  0. Grayscale + robust percentile normalize
  1. Median 3x3 (dead/hot pixel removal)
  2. Anisotropic Diffusion — Perona-Malik (grain removal, edge sharpening)
  3. Multi-Scale Retinex (contrast recovery from faded/washed-out images)
  4. Retinex + Diffusion blend (65/35)
  5. Adaptive NLM denoising (h adaptive, was fixed h=15)
  6. Morphological top-hat (lift small warm targets)
  7. CLAHE clip=3.0 (local contrast)
  8. Bilateral filter (edge-preserving CLAHE artifact cleanup)
  9. Unsharp masking amount=0.8 (crisp edges, no halos)
  10. IR gamma 0.80 (perceptual mapping)

Output: uint8 BGR (3-channel) for downstream YOLO detector compatibility.
"""

import os
import sys
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Import core v3.0 pipeline from parent src/ directory ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from enhance import (
    enhance_frame       as _core_enhance,
    detect_image_type   as _core_detect_type,
    robust_normalize,
    anisotropic_diffusion,
    multi_scale_retinex,
    apply_clahe,
    bilateral_denoise,
    tophat_enhance,
    unsharp_mask,
    ir_gamma,
    inpaint_artifacts,
    super_resolve,
)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (same signatures as before — backward compatible)
# ══════════════════════════════════════════════════════════════════════════════

def detect_image_type(frame: np.ndarray) -> dict:
    """
    Characterise the input frame.
    Wraps src/enhance.py's detect_image_type — no longer depends on skimage.
    """
    return _core_detect_type(frame)


def enhance_frame(frame: np.ndarray,
                  do_sr: bool = False,
                  sr_model_path: str = None,
                  return_8bit: bool = True,
                  params: dict = None) -> np.ndarray:
    """
    Full enhancement pipeline — v3.0.

    Delegates to src/enhance.py's core pipeline, then wraps the
    grayscale output into BGR for YOLO / Streamlit display compatibility.

    Input  : any frame (any dtype, grayscale or color)
    Output : uint8 BGR (3-channel) enhanced image

    Key improvements over the previous pipeline/enhance.py:
    ─────────────────────────────────────────────────────────
    REMOVED (were causing problems):
      ✗  Laplacian boost (strength=0.30)
         → Amplified noise along with edges. Responsible for the
           "noisy texture" look on enhanced film images.
      ✗  Aggressive unsharp mask (strength=1.8)
         → Was creating visible halos around vehicle boundaries.
         → Reduced to 0.8 in new pipeline.
      ✗  Fixed NLM h=15
         → Was over-smoothing everything, destroying weak edges
           in degraded film images. Now adaptive (h=4..12).

    ADDED (now active):
      ✓  Anisotropic Diffusion (Perona-Malik, 20 iterations)
         → Smooths film grain in flat areas, sharpens edges simultaneously.
         → Best single algorithm for old-film degradation noise.
      ✓  Multi-Scale Retinex (sigmas=[10, 60, 180])
         → Removes slow illumination gradient from faded film images.
         → Objects pop out even in globally washed-out scenes.
      ✓  Bilateral filter (after CLAHE)
         → Cleans up CLAHE blocking artifacts while keeping edges sharp.
      ✓  CLAHE clip_limit 2.0 → 3.0
         → Stronger local contrast, safe after diffusion cleaned background.
    """
    # Run the core v3.0 grayscale pipeline
    gray_enhanced = _core_enhance(
        frame,
        do_sr=do_sr,
        sr_model_path=sr_model_path,
        return_8bit=True,
        params=params,
    )

    # Ensure it's 2D grayscale
    if gray_enhanced.ndim == 3:
        gray_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_BGR2GRAY)

    # Convert grayscale → BGR for YOLO detector + Streamlit display
    bgr = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)

    return bgr if return_8bit else bgr


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY HELPERS — kept for any direct references in the codebase
# ══════════════════════════════════════════════════════════════════════════════

def normalise_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper → robust_normalize from core pipeline."""
    f = frame.astype(np.float32)
    if f.ndim == 3 and f.shape[2] >= 3:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    else:
        gray = f.squeeze()
    norm = robust_normalize(gray, lo_pct=0.5, hi_pct=99.5)
    out  = (norm * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def auto_gamma(bgr: np.ndarray) -> np.ndarray:
    """
    Backward-compatible auto gamma.
    Note: the main pipeline now uses fixed gamma=0.80 (better for film images).
    """
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val < 1:
        return bgr
    gamma    = np.log(mean_val / 255.0 + 1e-8) / np.log(0.5)
    gamma    = float(np.clip(gamma, 0.35, 2.5))
    inv_g    = 1.0 / gamma
    lut      = np.array([((i / 255.0) ** inv_g) * 255
                          for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, lut)


def clahe_enhance(bgr: np.ndarray,
                  clip_limit: float = 3.0,
                  tile: int = 8) -> np.ndarray:
    """Backward-compatible CLAHE on LAB L-channel."""
    lab          = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b      = cv2.split(lab)
    clahe_obj    = cv2.createCLAHE(clipLimit=clip_limit,
                                    tileGridSize=(tile, tile))
    lab_eq       = cv2.merge([clahe_obj.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def inpaint_dead_pixels(bgr: np.ndarray) -> np.ndarray:
    """Backward-compatible dead-pixel inpainter."""
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_v   = np.mean(gray)
    std_v    = np.std(gray)
    lo       = max(0,   mean_v - 5 * std_v)
    hi       = min(255, mean_v + 5 * std_v)
    mask     = ((gray < lo) | (gray > hi)).astype(np.uint8) * 255
    if np.sum(mask) == 0:
        return bgr
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, k, iterations=1)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)

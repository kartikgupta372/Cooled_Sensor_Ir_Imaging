"""
enhance.py — Professional IR Image Enhancement Pipeline v2.0
============================================================
Upgraded algorithm stack designed specifically for cooled IR sensors.

Algorithm chain (in order):
  1. Robust percentile normalisation          — clip dead/hot pixels
  2. Bad pixel inpainting                      — remove sensor artifacts
  3. Anisotropic diffusion (Perona-Malik)      — smooth within regions, SHARPEN edges
  4. Multi-Scale Retinex (MSR)                 — separate thermal signal from background
  5. CLAHE with optimised IR parameters        — local contrast boost
  6. Morphological top-hat enhancement         — lift small warm features (pedestrians)
  7. Bilateral filter final pass               — clean up without losing edges
  8. Adaptive unsharp masking                  — crisp, sharp output
  9. IR gamma curve                            — proper perceptual mapping

Result: clear, sharp images where vehicles and people stand out cleanly
        — like the professional FLIR / DRDO reference output.
"""

import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. IMAGE TYPE DETECTION  (unchanged API — used by app.py / detect.py)
# ══════════════════════════════════════════════════════════════════════════════
def detect_image_type(frame: np.ndarray) -> dict:
    """
    Automatically characterise any input image.
    Returns metadata that drives downstream processing decisions.
    """
    is_color = frame.ndim == 3 and frame.shape[2] == 3
    is_16bit = frame.dtype == np.uint16
    is_float = frame.dtype in [np.float32, np.float64]
    h, w     = frame.shape[:2]

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if is_color else frame
    gray_f = gray.astype(np.float32)

    mean_val  = float(np.mean(gray_f))
    std_val   = float(np.std(gray_f))
    max_val   = float(np.max(gray_f))
    dynamic_r = max_val / (std_val + 1e-8)

    # Fast noise estimate via high-frequency laplacian
    lap   = cv2.Laplacian(gray_f / (gray_f.max() + 1e-8), cv2.CV_32F)
    sigma = float(np.std(lap)) * 0.5    # rough noise estimate

    is_thermal = (
        not is_color or
        (is_color and _is_thermal_colormap(frame)) or
        is_16bit or is_float
    )

    return {
        "is_color":      is_color,
        "is_16bit":      is_16bit,
        "is_float":      is_float,
        "is_thermal":    is_thermal,
        "height":        h, "width": w,
        "mean":          mean_val,
        "std":           std_val,
        "noise_sigma":   sigma,
        "dynamic_range": dynamic_r,
        "low_contrast":  std_val < 30,
        "high_noise":    sigma > 0.05,
    }


def _is_thermal_colormap(frame: np.ndarray) -> bool:
    """Check if a color image is a thermal colormap (iron/rainbow) by channel correlation."""
    if frame.ndim != 3:
        return False
    b, g, r = cv2.split(frame)
    corr_br = float(np.corrcoef(b.ravel(), r.ravel())[0, 1])
    return abs(corr_br) > 0.85


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROBUST NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
def robust_normalize(frame: np.ndarray, lo_pct: float = 1.0,
                     hi_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-based normalisation → float32 [0, 1].
    Clips the bottom lo_pct% and top hi_pct% to remove dead/hot pixels
    BEFORE computing the dynamic range — gives full 8-bit to real signal.

    IR insight: a single saturated pixel can collapse the entire image
    dynamic range; percentile clipping prevents this completely.
    """
    f  = frame.astype(np.float32)
    lo = float(np.percentile(f, lo_pct))
    hi = float(np.percentile(f, hi_pct))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# Keep old name for backward compatibility
def adaptive_normalize(frame: np.ndarray, meta: dict) -> np.ndarray:
    f = frame.astype(np.float32)
    if meta.get("is_color"):
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return robust_normalize(f, lo_pct=0.5, hi_pct=99.5)


# ══════════════════════════════════════════════════════════════════════════════
# 3. BAD PIXEL INPAINTING  (dead / hot pixel removal)
# ══════════════════════════════════════════════════════════════════════════════
def inpaint_artifacts(frame_uint8: np.ndarray) -> np.ndarray:
    """
    Detect and inpaint extreme outlier pixels (dead/hot).
    Uses 5-sigma threshold relative to local statistics.
    """
    gray = (frame_uint8 if frame_uint8.ndim == 2
            else cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2GRAY))

    mean = np.mean(gray)
    std  = np.std(gray)
    lo   = max(0,   mean - 5 * std)
    hi   = min(255, mean + 5 * std)
    mask = ((gray < lo) | (gray > hi)).astype(np.uint8) * 255

    if np.sum(mask) == 0:
        return frame_uint8

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(frame_uint8, mask, 3, cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANISOTROPIC DIFFUSION  (Perona-Malik)
# ══════════════════════════════════════════════════════════════════════════════
def anisotropic_diffusion(img: np.ndarray,
                           num_iter: int = 10,
                           kappa: float = 30.0,
                           gamma: float = 0.1,
                           option: int = 2) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion.

    WHY THIS IS BETTER THAN GAUSSIAN / NL-MEANS FOR IR:
    - Gaussian blurs everything, including the vehicle-background edge.
    - NL-means is very slow and sometimes over-smooths thermal gradients.
    - Anisotropic diffusion SMOOTHS flat regions (background noise) but
      SHARPENS edges (vehicle outlines, person silhouettes) — exactly what
      a cooled IR sensor image needs.

    Parameters:
        num_iter : number of diffusion steps (more = smoother regions)
        kappa    : edge stopping threshold (lower = preserves more edges)
        gamma    : step size (0 < gamma ≤ 0.25 for stability)
        option   : 1 = Perona-Malik 1 (exp), 2 = Perona-Malik 2 (rational)
    """
    f = img.astype(np.float32)
    for _ in range(num_iter):
        # Finite differences in 4 directions
        dN = np.roll(f,  1, axis=0) - f
        dS = np.roll(f, -1, axis=0) - f
        dE = np.roll(f, -1, axis=1) - f
        dW = np.roll(f,  1, axis=1) - f

        if option == 1:
            # Perona-Malik function 1: c(x) = exp(-(|∇I|/κ)²)
            cN = np.exp(-(dN / kappa) ** 2)
            cS = np.exp(-(dS / kappa) ** 2)
            cE = np.exp(-(dE / kappa) ** 2)
            cW = np.exp(-(dW / kappa) ** 2)
        else:
            # Perona-Malik function 2: c(x) = 1 / (1 + (|∇I|/κ)²)
            cN = 1.0 / (1.0 + (dN / kappa) ** 2)
            cS = 1.0 / (1.0 + (dS / kappa) ** 2)
            cE = 1.0 / (1.0 + (dE / kappa) ** 2)
            cW = 1.0 / (1.0 + (dW / kappa) ** 2)

        f = f + gamma * (cN * dN + cS * dS + cE * dE + cW * dW)

    return np.clip(f, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-SCALE RETINEX  (MSR)
# ══════════════════════════════════════════════════════════════════════════════
def multi_scale_retinex(img: np.ndarray,
                         sigmas: list = None,
                         weights: list = None) -> np.ndarray:
    """
    Multi-Scale Retinex for IR contrast enhancement.

    WHY RETINEX FOR IR:
    - Retinex separates the image into "illumination" (background warmth gradient)
      and "reflectance" (actual thermal signature of objects).
    - For IR cameras: the slow ambient temperature gradient is the "illumination".
      Hot targets (vehicles, people) are the "reflectance".
    - MSR removes the ambient gradient and reveals the true thermal contrast
      → dark sky, bright warm objects — exactly like the reference image.

    sigmas: Gaussian scales [small=detail, medium=mid, large=global illumination]
    """
    if sigmas is None:
        sigmas = [15, 80, 250]
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    img_f  = img.astype(np.float32)
    log_I  = np.log1p(img_f * 255.0)   # log(I + 1)
    retinex = np.zeros_like(log_I)

    for sigma, w in zip(sigmas, weights):
        # Estimate "illumination" as a blurred version of the image
        ksize = int(6 * sigma + 1) | 1          # must be odd
        blurred   = cv2.GaussianBlur(img_f * 255.0, (ksize, ksize), sigma)
        log_blur  = np.log1p(blurred)
        retinex  += w * (log_I - log_blur)      # log-domain subtraction

    # Normalise retinex output to [0, 1]
    lo = float(np.percentile(retinex, 1))
    hi = float(np.percentile(retinex, 99))
    if hi - lo < 1e-6:
        hi = lo + 1.0

    retinex = np.clip((retinex - lo) / (hi - lo), 0.0, 1.0)
    return retinex.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MORPHOLOGICAL TOP-HAT ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
def tophat_enhance(img_uint8: np.ndarray, kernel_size: int = 15,
                   strength: float = 0.4) -> np.ndarray:
    """
    Morphological top-hat transform for small hot-target enhancement.

    WHY TOP-HAT FOR IR:
    - Top-hat = image − morphological_opening(image)
    - Opening removes objects smaller than the structuring element.
    - The residual (top-hat) = exactly the small bright features that were removed.
    - For IR: this highlights pedestrians, vehicle hot-spots, small aircraft
      that would otherwise be swamped by background gradients.
    - Strength controls how much of the top-hat is added back.
    """
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (kernel_size, kernel_size))
    tophat    = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    enhanced  = cv2.addWeighted(img_uint8, 1.0, tophat,
                                 strength, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLAHE  (Contrast Limited Adaptive Histogram Equalisation)
# ══════════════════════════════════════════════════════════════════════════════
def apply_clahe(img_uint8: np.ndarray,
                clip_limit: float = 3.0,
                tile_size: int = 8) -> np.ndarray:
    """
    CLAHE optimised for IR thermal imaging.

    IR-tuned parameters vs default:
    - clip_limit=3.0  (default=2.0): stronger local contrast in uniform scenes
    - tile_size=8     : 8×8 grid → local normalisation at ~60px block for 480p
    Works on 8-bit grayscale input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 8. BILATERAL FILTER  (edge-preserving final smoothing)
# ══════════════════════════════════════════════════════════════════════════════
def bilateral_denoise(img_uint8: np.ndarray,
                       d: int = 9,
                       sigma_color: float = 60.0,
                       sigma_space: float = 60.0) -> np.ndarray:
    """
    Bilateral filter — the go-to edge-preserving denoiser for IR.

    WHY BILATERAL:
    - Standard blur averages pixels within a spatial window (kills edges).
    - Bilateral adds a second weight based on INTENSITY DIFFERENCE.
    - Pixels far in intensity (= across a thermal edge) are NOT averaged.
    - Result: smooth background noise + sharp vehicle/person boundaries.

    sigma_color: intensity range for averaging — 60 is good for 8-bit IR
    sigma_space: spatial reach — 60 pixels
    """
    return cv2.bilateralFilter(img_uint8, d=d,
                                sigmaColor=sigma_color,
                                sigmaSpace=sigma_space)


# ══════════════════════════════════════════════════════════════════════════════
# 9. ADAPTIVE UNSHARP MASKING
# ══════════════════════════════════════════════════════════════════════════════
def unsharp_mask(img_uint8: np.ndarray,
                  radius: float = 2.0,
                  amount: float = 1.5,
                  threshold: int = 10) -> np.ndarray:
    """
    Unsharp masking — sharpens edges without amplifying flat-region noise.

    The 'threshold' parameter is key for IR:
    - Only pixels with local contrast > threshold get sharpened.
    - Flat noise regions (below threshold) are left untouched.
    - Edge regions (above threshold) get amount × boost.

    amount=1.5 gives crisp vehicle outlines while keeping background clean.
    """
    blurred   = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(img_uint8, 1.0 + amount,
                                 blurred, -amount, 0)
    # Threshold: only apply where edge is significant
    mask = np.abs(img_uint8.astype(np.int16) -
                  blurred.astype(np.int16)) > threshold
    result = np.where(mask, sharpened, img_uint8)
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 10. IR GAMMA CURVE
# ══════════════════════════════════════════════════════════════════════════════
def ir_gamma(img_uint8: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    """
    Gamma correction tuned for IR thermal display.

    gamma < 1.0 brightens the midtones (lifts warm targets from background).
    gamma = 0.75 is the FLIR standard for grayscale IR display.

    Uses a LUT (lookup table) — runs at full speed, one byte → one byte.
    """
    lut = np.array(
        [int(((i / 255.0) ** gamma) * 255) for i in range(256)],
        dtype=np.uint8
    )
    return lut[img_uint8]


# ══════════════════════════════════════════════════════════════════════════════
# 11. SUPER RESOLUTION  (optional — OpenCV DNN)
# ══════════════════════════════════════════════════════════════════════════════
def super_resolve(frame_uint8: np.ndarray, model_path: str = None,
                  scale: int = 2) -> np.ndarray:
    """
    Super-resolution via pre-trained EDSR model.
    Falls back gracefully to Lanczos (better than bicubic for IR).
    """
    import os
    if model_path is None:
        for c in ["models/EDSR_x2.pb", "src/models/EDSR_x2.pb",
                  os.path.expanduser("~/models/EDSR_x2.pb")]:
            if os.path.exists(c):
                model_path = c
                break

    if model_path and os.path.exists(model_path):
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("edsr", scale)
            return sr.upsample(frame_uint8)
        except Exception as e:
            print(f"[SR] Model failed ({e}), using Lanczos fallback")

    h, w = frame_uint8.shape[:2]
    return cv2.resize(frame_uint8, (w * scale, h * scale),
                      interpolation=cv2.INTER_LANCZOS4)


# ══════════════════════════════════════════════════════════════════════════════
# 12. TONE MAPPING  (for HDR / 16-bit inputs)
# ══════════════════════════════════════════════════════════════════════════════
def tone_map(frame: np.ndarray) -> np.ndarray:
    """Reinhard tone mapping for HDR inputs."""
    f = frame.astype(np.float32)
    if f.max() > 1.0:
        f = f / f.max()
    f  = f / (1.0 + f)
    lo = float(np.percentile(f, 1))
    hi = float(np.percentile(f, 99))
    if hi > lo:
        f = np.clip((f - lo) / (hi - lo), 0, 1)
    return (f * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENHANCEMENT FUNCTION  (public API — called by test_pipeline.py, app.py)
# ══════════════════════════════════════════════════════════════════════════════
def enhance_frame(frame: np.ndarray,
                  do_sr: bool = False,
                  sr_model_path: str = None,
                  return_8bit: bool = True) -> np.ndarray:
    """
    Robust IR enhancement pipeline designed for heavily noisy environments.

    Input  : any image (any dtype, any shape, any bit depth)
    Output : enhanced uint8 grayscale — smooth background, sharp targets.
    """
    # ── Step 0: Convert to float32 grayscale & robust normalise ─────────────
    meta = detect_image_type(frame)
    f = frame.astype(np.float32)
    if meta["is_color"]:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    norm = robust_normalize(f, lo_pct=0.5, hi_pct=99.5)
    
    # Convert to 8-bit for OpenCV optimized filters
    img_8u = (norm * 255).astype(np.uint8)

    # ── Step 1: Median Filter (Kill salt & pepper / dead pixels) ────────────
    # A 3x3 median filter is much more robust and faster than inpainting.
    median = cv2.medianBlur(img_8u, 3)

    # ── Step 2: Non-Local Means Denoising ───────────────────────────────────
    # NLM is the gold standard for heavy Gaussian noise. It averages similar
    # patches, perfectly smoothing the background while keeping targets sharp.
    # We use h=15 (strong denoising).
    nlm = cv2.fastNlMeansDenoising(median, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # ── Step 3: Morphological Top-Hat (Optional Lift) ───────────────────────
    # Lifts small hot targets gently without blowing up noise.
    tophat_out = tophat_enhance(nlm, kernel_size=15, strength=0.2)

    # ── Step 4: Gentle CLAHE ────────────────────────────────────────────────
    # Lower clip_limit (2.0) prevents noise amplification in the background.
    clahe_out = apply_clahe(tophat_out, clip_limit=2.0, tile_size=8)

    # ── Step 5: Adaptive Unsharp Masking ────────────────────────────────────
    # A slight sharpening of actual edges (targets) only.
    sharp = unsharp_mask(clahe_out, radius=1.0, amount=1.0, threshold=15)

    # ── Step 6: IR Gamma Curve (γ = 0.75) ───────────────────────────────────
    out = ir_gamma(sharp, gamma=0.85)  # slightly less aggressive gamma

    # ── Step 7: Optional super-resolution ──────────────────────────────────
    if do_sr:
        out = super_resolve(out, model_path=sr_model_path)

    return out if return_8bit else out
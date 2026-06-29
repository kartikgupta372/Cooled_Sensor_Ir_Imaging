"""
enhance.py — Professional IR Image Enhancement Pipeline v3.0
============================================================
Upgraded algorithm stack designed specifically for degraded / old-film
style images AND cooled IR sensors.

WHAT CHANGED FROM v2.0:
  ✅  Anisotropic Diffusion (Perona-Malik) — NOW ACTIVE in pipeline
      Previously defined but never called. This is the single most
      important fix for old-film / film-grain noise.

  ✅  Multi-Scale Retinex (MSR) — NOW ACTIVE in pipeline
      Previously defined but never called. Critical for fixing
      washed-out, low-contrast faded images.

  ✅  Bilateral filter — NOW ACTIVE in pipeline (was defined, not used)

  ✅  Adaptive NLM strength — h now scales with measured noise (was fixed h=15)
      h=15 was destroying edges and giving plastic "over-smoothed" output.

  ✅  CLAHE clip_limit 2.0 → 3.0 — stronger local contrast for film images

Algorithm chain (in order):
  0. Grayscale + robust percentile normalisation   → float32 [0, 1]
  1. Median 3×3                                    → kill dead/hot pixels
  2. Anisotropic Diffusion (Perona-Malik)          → smooth grain, SHARPEN edges
  3. Multi-Scale Retinex (MSR)                     → recover washed-out contrast
  4. Retinex + Diffusion blend (65/35)             → optimal depth retention
  5. Adaptive NLM denoising (h ∝ noise level)     → clean residual noise
  6. Morphological top-hat enhancement             → lift small warm targets
  7. CLAHE (clip=3.0)                              → strong local contrast
  8. Bilateral filter                              → edge-preserving cleanup
  9. Adaptive unsharp masking                      → crisp edges
  10. IR gamma curve (γ=0.80)                      → proper perceptual mapping
  11. (Optional) Super-resolution                  → upscale with EDSR
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
# 4. ANISOTROPIC DIFFUSION  (Perona-Malik)  ← KEY ALGORITHM FOR FILM GRAIN
# ══════════════════════════════════════════════════════════════════════════════
def anisotropic_diffusion(img: np.ndarray,
                           num_iter: int = 20,
                           kappa: float = 25.0,
                           gamma: float = 0.12,
                           option: int = 2) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion.

    WHY THIS IS THE BEST CHOICE FOR OLD-FILM / DEGRADED IMAGES:
    ─────────────────────────────────────────────────────────────
    - Gaussian blur: smooths everything uniformly → destroys edges.
    - Median filter: salt-and-pepper only, not film grain.
    - NLM: good but treats all texture similarly; over-smooths weak edges.

    Perona-Malik SOLVES this by being content-aware:
      c(∇I) = 1 / (1 + (|∇I| / κ)²)   [option 2, Perona-Malik 2]

    At a FLAT region (road, sky, background): |∇I| ≈ 0 → c ≈ 1 → full diffusion
    At an EDGE (vehicle outline, person): |∇I| >> κ → c ≈ 0 → NO diffusion

    Result: background noise smoothed away; object boundaries SHARPENED.
    This is exactly what you need for film-extracted noisy images.

    Parameters:
        num_iter : diffusion steps (20 = strong grain removal without edge blur)
        kappa    : edge stopping value. 25 = good for weak edges in degraded film.
                   Lower → more edges preserved. Higher → more smoothing.
        gamma    : step size (< 0.25 for numerical stability, 0.12 = safe + fast)
        option   : 2 = Perona-Malik 2 (better for noisy images than option 1)
    """
    f = img.astype(np.float32)
    for _ in range(num_iter):
        # Four-directional finite differences
        dN = np.roll(f,  1, axis=0) - f
        dS = np.roll(f, -1, axis=0) - f
        dE = np.roll(f, -1, axis=1) - f
        dW = np.roll(f,  1, axis=1) - f

        if option == 1:
            # PM1: c(x) = exp(-(|∇I|/κ)²)  — more aggressive edge stopping
            cN = np.exp(-(dN / kappa) ** 2)
            cS = np.exp(-(dS / kappa) ** 2)
            cE = np.exp(-(dE / kappa) ** 2)
            cW = np.exp(-(dW / kappa) ** 2)
        else:
            # PM2: c(x) = 1 / (1 + (|∇I|/κ)²)  — smoother, better for noisy images
            cN = 1.0 / (1.0 + (dN / kappa) ** 2)
            cS = 1.0 / (1.0 + (dS / kappa) ** 2)
            cE = 1.0 / (1.0 + (dE / kappa) ** 2)
            cW = 1.0 / (1.0 + (dW / kappa) ** 2)

        f = f + gamma * (cN * dN + cS * dS + cE * dE + cW * dW)

    return np.clip(f, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-SCALE RETINEX  (MSR)  ← KEY ALGORITHM FOR LOW-CONTRAST RECOVERY
# ══════════════════════════════════════════════════════════════════════════════
def multi_scale_retinex(img: np.ndarray,
                         sigmas: list = None,
                         weights: list = None) -> np.ndarray:
    """
    Multi-Scale Retinex for recovering contrast from degraded/faded images.

    WHY RETINEX IS CRITICAL FOR OLD-FILM / LOW-CONTRAST IR:
    ─────────────────────────────────────────────────────────
    The Retinex theory (Edwin Land, 1977) separates an image into:
        I(x,y) = L(x,y) × R(x,y)
    where:
        L = illumination  (slow, large-scale variation)
        R = reflectance   (fast, fine-scale — actual object detail)

    For OLD FILM images:
        L = uneven film development, fading, grain background
        R = actual scene content (vehicles, people, objects)

    For IR images:
        L = ambient temperature gradient (warm ground, cold sky)
        R = true thermal signature of hot targets

    MSR computes: log(R) = log(I) - weighted_sum(log(Gaussian_blur(I)))
    → removes the illumination component entirely
    → objects pop out even in globally washed-out, low-contrast images

    sigmas: [small, medium, large]
        small  (10-15) : recovers fine detail
        medium (60-80) : mid-range contrast
        large (180-250): removes global illumination gradient
    """
    if sigmas is None:
        sigmas = [10, 60, 180]
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    img_f  = img.astype(np.float32)
    log_I  = np.log1p(img_f * 255.0)   # log(I + 1), working in 0-255 range
    retinex = np.zeros_like(log_I)

    for sigma, w in zip(sigmas, weights):
        ksize    = int(6 * sigma + 1) | 1          # kernel must be odd
        blurred  = cv2.GaussianBlur(img_f * 255.0, (ksize, ksize), sigma)
        log_blur = np.log1p(blurred)
        retinex += w * (log_I - log_blur)           # log-domain subtraction = division in linear

    # Normalise retinex output → [0, 1], robust to outliers
    lo = float(np.percentile(retinex, 1))
    hi = float(np.percentile(retinex, 99))
    if hi - lo < 1e-6:
        # Fallback: if retinex collapsed, return original normalised image
        return robust_normalize(img, lo_pct=1.0, hi_pct=99.0)

    retinex = np.clip((retinex - lo) / (hi - lo), 0.0, 1.0)
    return retinex.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MORPHOLOGICAL TOP-HAT ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
def tophat_enhance(img_uint8: np.ndarray, kernel_size: int = 13,
                   strength: float = 0.25) -> np.ndarray:
    """
    Morphological top-hat transform for small hot-target enhancement.

    Top-hat = image − morphological_opening(image)
    The residual = exactly the small bright features (pedestrians, vehicles)
    that are smaller than the structuring element.
    Strength controls how aggressively they are boosted.
    """
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (kernel_size, kernel_size))
    tophat   = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.addWeighted(img_uint8, 1.0, tophat, strength, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLAHE  (Contrast Limited Adaptive Histogram Equalisation)
# ══════════════════════════════════════════════════════════════════════════════
def apply_clahe(img_uint8: np.ndarray,
                clip_limit: float = 3.0,
                tile_size: int = 8) -> np.ndarray:
    """
    CLAHE optimised for IR thermal / degraded film imaging.

    clip_limit=3.0 (up from 2.0):
        For heavily degraded film images, we need stronger local
        contrast to pull objects out of the washed background.
        3.0 gives more pop without over-amplifying noise
        (which is already handled by anisotropic diffusion before this step).

    tile_size=8:
        8×8 grid ensures localised enhancement for uneven illumination.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 8. BILATERAL FILTER  (edge-preserving final smoothing)  ← NOW ACTIVE
# ══════════════════════════════════════════════════════════════════════════════
def bilateral_denoise(img_uint8: np.ndarray,
                       d: int = 7,
                       sigma_color: float = 50.0,
                       sigma_space: float = 50.0) -> np.ndarray:
    """
    Bilateral filter — edge-preserving denoiser, now active in main pipeline.

    PREVIOUSLY: This function was defined but never called in enhance_frame().
    NOW: Called after CLAHE to remove any blocking/ringing artifacts CLAHE
         introduces, while preserving the sharp edges that diffusion + retinex
         worked hard to create.

    sigma_color=50: pixels within ±50 intensity units are averaged together
    sigma_space=50: spatial neighbourhood of ~50 pixels
    d=7: filter diameter (smaller than 9 for better speed, sufficient quality)
    """
    return cv2.bilateralFilter(img_uint8, d=d,
                                sigmaColor=sigma_color,
                                sigmaSpace=sigma_space)


# ══════════════════════════════════════════════════════════════════════════════
# 9. ADAPTIVE UNSHARP MASKING
# ══════════════════════════════════════════════════════════════════════════════
def unsharp_mask(img_uint8: np.ndarray,
                  radius: float = 1.5,
                  amount: float = 0.8,
                  threshold: int = 10) -> np.ndarray:
    """
    Unsharp masking — sharpens edges without amplifying flat-region noise.

    amount=0.8 (reduced from 1.5):
        After diffusion + retinex, edges are already well-defined.
        Aggressive unsharp masking (amount=1.5) was causing halo artifacts.
        0.8 gives crisp boundaries without halos.

    threshold=10:
        Only pixels with local contrast > 10 are sharpened.
        This protects the smoothed background from re-gaining noise.
    """
    blurred   = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(img_uint8, 1.0 + amount,
                                 blurred, -amount, 0)
    mask   = np.abs(img_uint8.astype(np.int16) -
                    blurred.astype(np.int16)) > threshold
    result = np.where(mask, sharpened, img_uint8)
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 10. IR GAMMA CURVE
# ══════════════════════════════════════════════════════════════════════════════
def ir_gamma(img_uint8: np.ndarray, gamma: float = 0.80) -> np.ndarray:
    """
    Gamma correction tuned for IR thermal display.

    gamma=0.80 (tweaked from 0.85):
        γ < 1.0 brightens midtones → warm targets lift from background.
        0.80 gives slightly more pop for faded film images.
        Uses a LUT for maximum speed (one byte → one byte mapping).
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
# MASTER ENHANCEMENT FUNCTION  (v3.0 — public API)
# ══════════════════════════════════════════════════════════════════════════════
def enhance_frame(frame: np.ndarray,
                  do_sr: bool = False,
                  sr_model_path: str = None,
                  return_8bit: bool = True,
                  params: dict = None) -> np.ndarray:
    """
    v3.0 IR Enhancement Pipeline — optimised for degraded / old-film images.

    Key changes from v2.0:
    ─────────────────────
    1. Anisotropic Diffusion (Perona-Malik) ADDED → best grain smoother
       that preserves and actually sharpens object edges.

    2. Multi-Scale Retinex ADDED → removes background illumination gradient
       and recovers contrast from faded / washed-out film images.

    3. Adaptive NLM h → was fixed h=15 (too aggressive, destroys edges).
       Now h scales with measured noise: h ∈ [4, 12].

    4. Bilateral filter ADDED (was defined, never called) → removes CLAHE
       blocking artifacts while keeping edges from step 2-3.

    5. CLAHE clip_limit 2.0 → 3.0 → stronger local contrast for film images.

    6. Unsharp amount 1.5 → 0.8 → prevents halo artifacts after Retinex.

    Input  : any image (any dtype, any shape, any bit depth)
    Output : enhanced uint8 grayscale — clean background, sharp targets.
    """
    p = params or {}

    # ── Step 0: Convert to float32 grayscale & robust normalise ─────────────
    meta = detect_image_type(frame)
    f = frame.astype(np.float32)
    if meta["is_color"]:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    norm = robust_normalize(f, lo_pct=0.5, hi_pct=99.5)   # float32 [0, 1]

    # ── Step 1: Median 3×3 — kill salt-and-pepper / dead pixels ─────────────
    # A fast 3×3 median is the most reliable dead/hot pixel killer.
    # Runs in uint8 space for speed, then converts back to float32.
    img_8u     = (norm * 255).astype(np.uint8)
    median_out = cv2.medianBlur(img_8u, 3)
    f_clean    = median_out.astype(np.float32) / 255.0

    # ── Step 2: Anisotropic Diffusion (Perona-Malik) ─────────────────────────
    # THE SINGLE MOST IMPORTANT FIX for film-grain / old-film images.
    #
    # What it does:
    #   • In flat regions (road, sky, background): diffuses heavily → grain removed
    #   • At edges (vehicle outline, person silhouette): stops completely → edge sharpened
    #
    # Why it beats Gaussian or NLM alone:
    #   • Gaussian: treats edges and flat areas equally → edges blurred
    #   • NLM: can over-smooth weak edges in degraded images (h=15 was catastrophic)
    #   • Perona-Malik: content-aware → smooth background, crisp foreground
    #
    # Parameters (tuned for noisy / degraded film images):
    #   num_iter=20 : strong grain removal in flat areas
    #   kappa=25    : low enough to preserve even weak film-degraded edges
    #   gamma=0.12  : safe step size (must be < 0.25)
    f_diffused = anisotropic_diffusion(
        f_clean,
        num_iter=p.get("anisotropic_iterations", 20),
        kappa=p.get("anisotropic_kappa", 25.0),
        gamma=p.get("anisotropic_gamma", 0.12),
        option=2
    )

    # ── Step 3: Multi-Scale Retinex (MSR) ────────────────────────────────────
    # THE SECOND KEY FIX for faded / low-contrast film images.
    #
    # What it does:
    #   • Separates illumination (fading, uneven development, background glow)
    #     from reflectance (actual vehicles, people, objects).
    #   • Removes the illumination component → objects pop out against background.
    #
    # Why it matters for old film:
    #   • Old film images are often globally washed out / grey everywhere.
    #   • Standard CLAHE boosts local contrast but can't fix the global problem.
    #   • Retinex removes the slow background component → reveals true detail.
    #
    # sigmas=[10, 60, 180] — three scales:
    #   10  : recovers fine grain-level detail
    #   60  : recovers mid-range object contrast
    #   180 : removes global background illumination gradient
    f_retinex = multi_scale_retinex(
        f_diffused,
        sigmas=p.get("retinex_sigmas", [10, 60, 180])
    )

    # Blend: 65% Retinex + 35% Diffused
    # Pure Retinex occasionally loses volumetric depth (objects look flat).
    # Blending keeps depth + still fixes the contrast problem.
    blend_ratio = p.get("retinex_blend", 0.65)
    f_blended   = blend_ratio * f_retinex + (1.0 - blend_ratio) * f_diffused
    f_blended   = np.clip(f_blended, 0.0, 1.0)

    # ── Step 4: Adaptive NLM Denoising ──────────────────────────────────────
    # After diffusion + retinex, the image is already much cleaner.
    # We use a NOISE-ADAPTIVE h instead of the previous fixed h=15.
    #
    # h=15 was the root cause of the "plastic / over-smoothed" output:
    # it was smoothing everything including the edges that diffusion had
    # just sharpened. Now h scales between 4 (clean) and 12 (very noisy).
    #
    # Formula: h = clip(noise_sigma × 80, 4, 12)
    # Example: noise_sigma=0.08 → h=6 (moderate denoising)
    #          noise_sigma=0.15 → h=12 (heavy denoising for very noisy input)
    noise_sigma = meta.get("noise_sigma", 0.05)
    h_nlm       = int(np.clip(noise_sigma * 80,
                               p.get("nlm_h_min", 4),
                               p.get("nlm_h_max", 12)))
    img_8u = (f_blended * 255).astype(np.uint8)
    nlm    = cv2.fastNlMeansDenoising(
        img_8u, None,
        h=h_nlm,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # ── Step 5: Morphological Top-Hat — lift small warm targets ─────────────
    # Highlights small bright features (pedestrians, vehicle hot-spots)
    # that might still be dim relative to their surroundings.
    # strength=0.25 is gentle — we don't want to re-introduce grain.
    tophat_out = tophat_enhance(
        nlm,
        kernel_size=p.get("tophat_kernel_size", 13),
        strength=p.get("tophat_strength", 0.25)
    )

    # ── Step 6: CLAHE — local contrast boost ────────────────────────────────
    # clip_limit=3.0 (increased from 2.0):
    #   After diffusion cleaned the background, CLAHE now boosts real contrast
    #   rather than amplifying noise. 3.0 is safe here.
    clahe_out = apply_clahe(
        tophat_out,
        clip_limit=p.get("clahe_clip_limit", 3.0),
        tile_size=p.get("clahe_tile_size", 8)
    )

    # ── Step 7: Bilateral Filter ─────────────────────────────────────────────
    # PREVIOUSLY DEFINED BUT NEVER CALLED. Now active.
    # Purpose: Remove any CLAHE blocking/tiling artifacts while keeping
    # the sharp vehicle/person edges that we worked hard to create.
    # sigma_color=50, sigma_space=50: conservative — we're mostly done.
    bilateral_out = bilateral_denoise(
        clahe_out,
        d=p.get("bilateral_d", 7),
        sigma_color=p.get("bilateral_sigma_color", 50.0),
        sigma_space=p.get("bilateral_sigma_space", 50.0)
    )

    # ── Step 8: Adaptive Unsharp Masking ────────────────────────────────────
    # amount=0.8 (reduced from 1.5):
    #   Retinex already sharpened the edges. We just need a light final pass.
    #   1.5 was causing halo artifacts around vehicle boundaries.
    sharp = unsharp_mask(
        bilateral_out,
        radius=p.get("unsharp_radius", 1.5),
        amount=p.get("unsharp_amount", 0.8),
        threshold=p.get("unsharp_threshold", 10)
    )

    # ── Step 9: IR Gamma Curve ───────────────────────────────────────────────
    # gamma=0.80 (slightly more aggressive than 0.85):
    #   Lifts the midtones more for faded film images where targets are dim.
    out = ir_gamma(sharp, gamma=p.get("gamma", 0.80))

    # ── Step 10: Optional Super-Resolution ──────────────────────────────────
    if do_sr:
        out = super_resolve(out, model_path=sr_model_path)

    return out if return_8bit else out

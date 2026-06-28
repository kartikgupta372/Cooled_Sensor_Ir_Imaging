import os
import json
import yaml
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_params(yaml_path="params.yaml"):
    """Loads our configuration from the central params.yaml file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def compute_psnr(original, processed, max_val):
    """
    Peak Signal-to-Noise Ratio (PSNR).
    
    Formula:  PSNR = 10 * log10(MAX^2 / MSE)
    
    Where:
        MAX = maximum possible pixel value (16383 for 14-bit)
        MSE = Mean Squared Error between original and processed images
    
    Higher PSNR = less noise = better quality.
    Typical IR values: 25-40 dB.
    """
    return peak_signal_noise_ratio(
        original.astype(np.float64),
        processed.astype(np.float64),
        data_range=max_val
    )

def compute_ssim(original, processed, max_val):
    """
    Structural Similarity Index (SSIM).
    
    Unlike PSNR (which just counts pixel-by-pixel differences), 
    SSIM looks at THREE things:
        1. Luminance  — are the overall brightness levels similar?
        2. Contrast   — are the ranges of bright/dark similar?
        3. Structure  — are the edges and textures preserved?
    
    Returns a value between -1 and 1. 
    1.0 = identical images, 0.0 = no similarity.
    """
    return structural_similarity(
        original.astype(np.float64),
        processed.astype(np.float64),
        data_range=max_val
    )

def evaluate_pipeline(raw_dir, processed_dir, max_val):
    """
    Compares every raw image against its processed counterpart.
    Returns the average PSNR and SSIM across all image pairs.
    """
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".tiff")])
    
    psnr_scores = []
    ssim_scores = []
    
    for filename in raw_files:
        raw_path = os.path.join(raw_dir, filename)
        proc_path = os.path.join(processed_dir, filename)
        
        # Skip if processed version doesn't exist
        if not os.path.exists(proc_path):
            continue
        
        raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        proc_img = cv2.imread(proc_path, cv2.IMREAD_UNCHANGED)
        
        psnr = compute_psnr(raw_img, proc_img, max_val)
        ssim = compute_ssim(raw_img, proc_img, max_val)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
    
    return {
        "avg_psnr": round(float(np.mean(psnr_scores)), 4),
        "avg_ssim": round(float(np.mean(ssim_scores)), 4),
        "num_images": len(psnr_scores)
    }

def main():
    params = load_params()
    
    raw_dir = params["simulate"]["output_dir"]          # data/raw
    processed_dir = params["process"]["output_dir"]      # data/processed/deterministic
    bit_depth = params["base"]["bit_depth"]
    max_val = (2 ** bit_depth) - 1                       # 16383
    
    output_file = params["evaluate"]["output_file"]      # metrics/evaluation.json
    
    print("Evaluating pipeline quality (PSNR & SSIM)...")
    
    metrics = evaluate_pipeline(raw_dir, processed_dir, max_val)
    
    # Save metrics as JSON (DVC can track this as a metric file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  PSNR: {metrics['avg_psnr']} dB")
    print(f"  SSIM: {metrics['avg_ssim']}")
    print(f"  Images evaluated: {metrics['num_images']}")
    print(f"  Saved to {output_file}")

if __name__ == "__main__":
    main()

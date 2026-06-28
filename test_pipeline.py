"""
test_pipeline.py — Run this to verify the entire fixed pipeline works.
Usage:
    python test_pipeline.py                        # uses built-in test image
    python test_pipeline.py --image path/to/any.png
    python test_pipeline.py --folder path/to/folder/
"""

import sys, os, argparse, time
sys.path.insert(0, "src")

import cv2
import numpy as np

def make_test_image():
    """Generate a synthetic test image when no real image is provided."""
    h, w = 480, 640
    # Gradient background (like a thermal scene)
    x = np.linspace(0, 255, w)
    y = np.linspace(100, 200, h)
    bg = np.outer(y, np.ones(w)).astype(np.float32)
    # Add some 'objects' (bright blobs like vehicles/hotspots)
    cv2.circle(bg, (160, 240), 40, 240, -1)   # vehicle 1
    cv2.circle(bg, (400, 180), 30, 220, -1)   # vehicle 2
    cv2.rectangle(bg, (500, 300), (580, 400), 235, -1)  # building hotspot
    # Add noise
    noise = np.random.normal(0, 15, bg.shape).astype(np.float32)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return bg


def test_single_image(image_path=None):
    from pipeline.ingest       import load_frame, validate_frame, get_frame_info
    from pipeline.calibrate    import run_calibration, load_calibration
    from pipeline.deterministic import run_deterministic, KalmanFrameFilter, BackgroundModel
    from pipeline.enhance      import enhance_frame
    from pipeline.detect       import UnifiedDetector, draw_detections, detections_to_json

    print("\n" + "="*60)
    print("  IR PIPELINE v2.0 — FULL TEST")
    print("="*60)

    # ── Load image ────────────────────────────────────────────────────────────
    if image_path and os.path.exists(image_path):
        print(f"\n[1/5] Loading: {image_path}")
        frame = load_frame(image_path)
    else:
        print("\n[1/5] No image provided — using synthetic test image")
        frame = make_test_image().astype(np.float32)

    valid, reason = validate_frame(frame)
    assert valid, f"Invalid frame: {reason}"
    info = get_frame_info(frame)
    print(f"      Shape: {info['shape']}  dtype: {info['dtype']}  "
          f"range: [{info['min']:.1f}, {info['max']:.1f}]")

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\n[2/5] Calibration (auto-estimates if no calib files)...")
    t = time.perf_counter()
    calib     = load_calibration("data/calibration_assets")
    calibrated = run_calibration(frame, calib=calib)
    print(f"      Done in {(time.perf_counter()-t)*1000:.1f}ms")

    # ── Deterministic processing ──────────────────────────────────────────────
    print("\n[3/5] Deterministic processing...")
    t = time.perf_counter()
    kalman   = KalmanFrameFilter(shape=calibrated.shape[:2])
    bg_model = BackgroundModel(alpha=0.05)
    processed = run_deterministic(calibrated, kalman=kalman, bg_model=bg_model)
    print(f"      Done in {(time.perf_counter()-t)*1000:.1f}ms")

    # ── Enhancement ───────────────────────────────────────────────────────────
    print("\n[4/5] Enhancement (adaptive — no training data needed)...")
    t = time.perf_counter()
    enhanced = enhance_frame(processed, do_sr=False, return_8bit=True)
    print(f"      Done in {(time.perf_counter()-t)*1000:.1f}ms  "
          f"shape: {enhanced.shape}")

    # ── Detection ─────────────────────────────────────────────────────────────
    print("\n[5/5] Object detection (80 COCO classes + thermal)...")
    print("      Loading YOLOv8n (auto-downloads ~6MB on first run)...")
    t = time.perf_counter()
    detector   = UnifiedDetector(yolo_model_size="n", yolo_conf=0.25,
                                  use_thermal=True)
    detections = detector.detect(enhanced, is_thermal=True)
    elapsed    = (time.perf_counter()-t)*1000
    print(f"      Done in {elapsed:.1f}ms")
    print(f"      Detected {len(detections)} objects:")
    for d in detections:
        print(f"        [{d.source:8s}] {d.label:<20s} conf={d.confidence:.2f}  "
              f"bbox={d.bbox}  track#{d.track_id}")

    # ── Annotate & Save ───────────────────────────────────────────────────────
    annotated = draw_detections(enhanced, detections)
    os.makedirs("data/output", exist_ok=True)
    out_path = "data/output/test_output.png"
    cv2.imwrite(out_path, annotated)

    det_json = detections_to_json(detections, "test_frame")
    import json
    with open("data/output/test_detections.json", "w") as f:
        json.dump(det_json, f, indent=2)

    print(f"\n✅ Pipeline complete!")
    print(f"   Enhanced image : {out_path}")
    print(f"   Detections JSON: data/output/test_detections.json")
    print(f"   Total objects  : {len(detections)}")


def test_folder(folder_path):
    from pipeline.ingest import discover_folder, load_frame, validate_frame
    from pipeline.calibrate import run_calibration, load_calibration
    from pipeline.deterministic import run_deterministic, KalmanFrameFilter, BackgroundModel
    from pipeline.enhance import enhance_frame
    from pipeline.detect import UnifiedDetector, draw_detections, detections_to_json
    import json

    print(f"\nBatch processing: {folder_path}")
    paths = discover_folder(folder_path)
    print(f"Found {len(paths)} images\n")

    calib     = load_calibration("data/calibration_assets")
    detector  = UnifiedDetector(yolo_model_size="n", yolo_conf=0.25)
    os.makedirs("data/output/batch", exist_ok=True)

    kalman   = None
    bg_model = BackgroundModel(alpha=0.05)

    for i, path in enumerate(paths):
        try:
            frame  = load_frame(path)
            valid, _ = validate_frame(frame)
            if not valid:
                continue
            if kalman is None:
                kalman = KalmanFrameFilter(shape=frame.shape[:2])
            cal    = run_calibration(frame, calib=calib)
            proc   = run_deterministic(cal, kalman=kalman, bg_model=bg_model)
            enh    = enhance_frame(proc, do_sr=False, return_8bit=True)
            dets   = detector.detect(enh, is_thermal=True)
            ann    = draw_detections(enh, dets)
            name   = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(f"data/output/batch/{name}_enhanced.png", ann)
            print(f"[{i+1}/{len(paths)}] {name}: {len(dets)} detections")
        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    print("\n✅ Batch complete → data/output/batch/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    args = parser.parse_args()

    if args.folder:
        test_folder(args.folder)
    else:
        test_single_image(args.image)
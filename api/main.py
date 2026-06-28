"""
api/main.py — FastAPI pipeline server.
POST /process        → single image
POST /process/batch  → folder of images (async)
GET  /job/{id}       → poll batch status
GET  /health         → health check
"""

import os, sys, uuid, json, time, io, asyncio, traceback
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import yaml

# ── Add src to path ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.ingest      import load_frame, discover_folder, validate_frame
from pipeline.calibrate   import run_calibration, load_calibration
from pipeline.deterministic import (run_deterministic, KalmanFrameFilter,
                                     BackgroundModel)
from pipeline.enhance     import enhance_frame
from pipeline.detect      import UnifiedDetector, draw_detections, detections_to_json

# ── Load params ───────────────────────────────────────────────────────────────
PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
with open(PARAMS_PATH) as f:
    PARAMS = yaml.safe_load(f)

CALIB_DIR   = PARAMS.get("paths", {}).get("calibration_dir", "data/calibration_assets")
DO_SR       = PARAMS.get("enhance", {}).get("super_resolution", False)
SR_MODEL    = PARAMS.get("enhance", {}).get("sr_model_path", None)
YOLO_SIZE   = PARAMS.get("detection", {}).get("yolo_model_size", "n")
YOLO_CONF   = PARAMS.get("detection", {}).get("yolo_conf", 0.25)
DEVICE      = PARAMS.get("detection", {}).get("device", "cpu")

# ── Load calibration once at startup ──────────────────────────────────────────
CALIB = load_calibration(CALIB_DIR)

# ── Initialise detector (downloads YOLOv8n.pt on first run) ──────────────────
detector = UnifiedDetector(
    yolo_model_size=YOLO_SIZE,
    yolo_conf=YOLO_CONF,
    use_thermal=True,
    device=DEVICE,
)

# ── Job registry for batch tracking ──────────────────────────────────────────
JOBS: dict = {}           # job_id → status dict
executor   = ThreadPoolExecutor(max_workers=4)

app = FastAPI(
    title="IR Imaging Pipeline",
    version="2.0.0",
    description="Universal image enhancement + object detection pipeline"
)


# ═════════════════════════════════════════════════════════════════════════════
# CORE PROCESSING — called for every frame
# ═════════════════════════════════════════════════════════════════════════════
def process_single_frame(frame: np.ndarray,
                          frame_id: str = "frame",
                          kalman: KalmanFrameFilter = None,
                          bg_model: BackgroundModel = None) -> dict:
    """
    Full 5-stage pipeline on one frame.
    Returns dict with annotated image bytes + detection JSON.
    """
    t0 = time.perf_counter()

    # ── Stage 1: Calibration (auto-falls-back if no calib files) ─────────────
    calibrated = run_calibration(frame, calib=CALIB, calib_dir=CALIB_DIR)

    # ── Stage 2: Deterministic processing ────────────────────────────────────
    processed = run_deterministic(
        calibrated,
        kalman=kalman,
        bg_model=bg_model,
        params=PARAMS.get("deterministic", {})
    )

    # ── Stage 3 + 4: Enhancement (adaptive, works on any image) ──────────────
    enhanced = enhance_frame(
        processed,
        do_sr=DO_SR,
        sr_model_path=SR_MODEL,
        return_8bit=True
    )

    # ── Stage 5: Detection (80 COCO + thermal classes) ────────────────────────
    # Detect on enhanced image for best results
    meta = {"is_thermal": _is_thermal_like(frame)}
    detections = detector.detect(enhanced, is_thermal=meta["is_thermal"])

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = draw_detections(enhanced, detections)

    # ── Encode to PNG ─────────────────────────────────────────────────────────
    _, buf = cv2.imencode(".png", annotated)
    img_bytes = buf.tobytes()

    elapsed = (time.perf_counter() - t0) * 1000   # ms

    det_json = detections_to_json(detections, frame_id)
    det_json["processing_ms"] = round(elapsed, 1)

    return {
        "image_bytes": img_bytes,
        "detections":  det_json,
        "frame_id":    frame_id,
    }


def _is_thermal_like(frame: np.ndarray) -> bool:
    """Heuristic: 16-bit or float input → likely thermal."""
    return frame.dtype in (np.uint16, np.float32, np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Upload any single image → returns enhanced PNG + X-Detections header.
    Accepts: .tiff, .png, .jpg, .npy, .raw, .bin, .bmp, etc.
    """
    import tempfile
    content = await file.read()
    suffix  = Path(file.filename).suffix or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        frame = load_frame(tmp_path)
        valid, reason = validate_frame(frame)
        if not valid:
            raise HTTPException(status_code=422, detail=f"Invalid image: {reason}")

        result = process_single_frame(frame, frame_id=file.filename)

        return StreamingResponse(
            io.BytesIO(result["image_bytes"]),
            media_type="image/png",
            headers={
                "X-Detections":     str(result["detections"]["count"]),
                "X-Processing-Ms":  str(result["detections"]["processing_ms"]),
                "X-Frame-Id":       result["frame_id"],
                "X-Detection-JSON": json.dumps(result["detections"]),
            }
        )
    finally:
        os.unlink(tmp_path)


@app.post("/process/batch")
async def process_batch(background_tasks: BackgroundTasks,
                         folder_path: str = "",
                         output_dir: str = "data/output",
                         async_mode: bool = True):
    """
    Process an entire folder of images.
    async_mode=True  → returns job_id immediately, process in background.
    async_mode=False → blocks until complete (small batches only).
    """
    if not folder_path:
        raise HTTPException(status_code=422, detail="folder_path is required")

    try:
        paths = discover_folder(folder_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    JOBS[job_id] = {
        "status":      "queued",
        "total":       len(paths),
        "done":        0,
        "errors":      0,
        "output_dir":  output_dir,
        "folder_path": folder_path,
        "start_time":  time.time(),
        "results":     [],
    }

    if async_mode:
        background_tasks.add_task(_run_batch_job, job_id, paths, output_dir)
        return JSONResponse({
            "job_id":      job_id,
            "status":      "queued",
            "frame_count": len(paths),
            "poll_url":    f"/job/{job_id}",
        })
    else:
        # Synchronous — blocks
        await asyncio.get_event_loop().run_in_executor(
            executor, lambda: _run_batch_job_sync(job_id, paths, output_dir))
        return JSONResponse(JOBS[job_id])


def _run_batch_job(job_id: str, paths: list, output_dir: str):
    """Background task: process all frames, save output."""
    asyncio.run(_run_batch_job_async(job_id, paths, output_dir))


async def _run_batch_job_async(job_id: str, paths: list, output_dir: str):
    await asyncio.get_event_loop().run_in_executor(
        executor, lambda: _run_batch_job_sync(job_id, paths, output_dir))


def _run_batch_job_sync(job_id: str, paths: list, output_dir: str):
    job = JOBS[job_id]
    job["status"] = "running"
    os.makedirs(output_dir, exist_ok=True)

    # Stateful components shared across batch for temporal filtering
    sample_frame = load_frame(paths[0])
    kalman   = KalmanFrameFilter(shape=sample_frame.shape[:2])
    bg_model = BackgroundModel(alpha=0.05)

    for i, path in enumerate(paths):
        try:
            frame = load_frame(path)
            valid, reason = validate_frame(frame)
            if not valid:
                job["errors"] += 1
                continue

            frame_id = Path(path).stem
            result   = process_single_frame(
                frame, frame_id=frame_id,
                kalman=kalman, bg_model=bg_model
            )

            # Save enhanced image
            out_img  = os.path.join(output_dir, f"{frame_id}_enhanced.png")
            out_json = os.path.join(output_dir, f"{frame_id}_detections.json")
            with open(out_img, "wb") as f:
                f.write(result["image_bytes"])
            with open(out_json, "w") as f:
                json.dump(result["detections"], f, indent=2)

            job["results"].append({
                "frame_id":   frame_id,
                "detections": result["detections"]["count"],
                "output_img": out_img,
            })

        except Exception as e:
            job["errors"] += 1
            print(f"[BATCH] Error on {path}: {e}\n{traceback.format_exc()}")

        job["done"] += 1
        if i % 10 == 0:
            pct = job["done"] / job["total"] * 100
            print(f"[BATCH {job_id}] {pct:.1f}% ({job['done']}/{job['total']})")

    job["status"]   = "complete"
    job["elapsed_s"] = round(time.time() - job["start_time"], 1)
    print(f"[BATCH {job_id}] Done. {job['done']} frames, {job['errors']} errors.")


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    job  = JOBS[job_id]
    done = job["done"]
    total = job["total"]
    return JSONResponse({
        "job_id":      job_id,
        "status":      job["status"],
        "progress":    round(done / total, 3) if total else 0,
        "frames_done": done,
        "frames_total":total,
        "errors":      job["errors"],
        "output_dir":  job["output_dir"],
        "elapsed_s":   round(time.time() - job["start_time"], 1),
        "results_sample": job["results"][:5],   # first 5 as preview
    })


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "version":      "2.0.0",
        "yolo_loaded":  detector.yolo.model is not None,
        "calib_loaded": any(v is not None for v in CALIB.values()),
        "coco_classes": 80,
        "thermal_classes": 5,
    }


@app.get("/classes")
async def list_classes():
    """List all detectable object classes."""
    from pipeline.detect import COCO_CLASSES, IR_CLASSES
    return {
        "coco_classes":    COCO_CLASSES,
        "thermal_classes": list(IR_CLASSES.keys()),
        "total":           len(COCO_CLASSES) + len(IR_CLASSES),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=False, workers=1)
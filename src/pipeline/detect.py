# pipeline/detect.py
# Re-exports everything from src/detect.py
# This allows: from pipeline.detect import UnifiedDetector, draw_detections, ...

from detect import (
    COCO_CLASSES,
    IR_CLASSES,
    CLASS_COLORS,
    Detection,
    UniversalYOLODetector,
    ThermalHotspotDetector,
    CentroidTracker,
    UnifiedDetector,
    draw_detections,
    detections_to_json,
)

__all__ = [
    "COCO_CLASSES",
    "IR_CLASSES",
    "CLASS_COLORS",
    "Detection",
    "UniversalYOLODetector",
    "ThermalHotspotDetector",
    "CentroidTracker",
    "UnifiedDetector",
    "draw_detections",
    "detections_to_json",
]

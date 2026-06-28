"""
app.py — Streamlit frontend for the IR Imaging Pipeline
Run: streamlit run app.py
"""

import sys, os, io, json, time, zipfile, tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ── Add src to Python path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IR Imaging Pipeline",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — dark themed, custom design ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #060B14 !important;
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] {
    background-color: #0A0E1A !important;
    border-right: 1px solid #1E3A5F !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ─────────────────────────────────────────────────── */
h1, h2, h3, h4 { color: #F1F5F9 !important; letter-spacing: 0.5px; }
p, li, span, label { color: #CBD5E1 !important; }

/* ── Cards ──────────────────────────────────────────────────────── */
.ir-card {
    background: #111827;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.ir-card-accent-cyan  { border-top: 3px solid #00D4FF; }
.ir-card-accent-green { border-top: 3px solid #10B981; }
.ir-card-accent-yell  { border-top: 3px solid #F59E0B; }
.ir-card-accent-purp  { border-top: 3px solid #8B5CF6; }
.ir-card-accent-red   { border-top: 3px solid #EF4444; }

/* ── Metric tiles ───────────────────────────────────────────────── */
.metric-row { display:flex; gap:12px; margin:0.8rem 0; flex-wrap:wrap; }
.metric-tile {
    background:#111827;
    border:1px solid #1E3A5F;
    border-radius:10px;
    padding:12px 18px;
    min-width:110px;
    text-align:center;
    flex:1;
}
.metric-tile .val {
    font-size:1.6rem;
    font-weight:700;
    line-height:1.1;
    margin-bottom:3px;
}
.metric-tile .lbl {
    font-size:0.7rem;
    letter-spacing:1.5px;
    text-transform:uppercase;
    color:#64748B;
}

/* ── Badges ─────────────────────────────────────────────────────── */
.badge {
    display:inline-block;
    font-size:0.72rem;
    font-weight:600;
    padding:3px 10px;
    border-radius:20px;
    margin:2px 3px;
    letter-spacing:.4px;
}
.badge-cyan  { background:#00D4FF22; color:#00D4FF; border:1px solid #00D4FF44; }
.badge-green { background:#10B98122; color:#10B981; border:1px solid #10B98144; }
.badge-yell  { background:#F59E0B22; color:#F59E0B; border:1px solid #F59E0B44; }
.badge-red   { background:#EF444422; color:#EF4444; border:1px solid #EF444444; }
.badge-purp  { background:#8B5CF622; color:#8B5CF6; border:1px solid #8B5CF644; }
.badge-blue  { background:#3B82F622; color:#3B82F6; border:1px solid #3B82F644; }

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton>button {
    background: #0A1628 !important;
    color: #00D4FF !important;
    border: 1px solid #00D4FF44 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .5px !important;
    transition: all .2s !important;
}
.stButton>button:hover {
    background: #00D4FF !important;
    color: #060B14 !important;
    border-color: #00D4FF !important;
}

/* ── File uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1.5px dashed #1E3A5F !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00D4FF !important;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0A0E1A !important;
    border-bottom: 1px solid #1E3A5F !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: #111827 !important;
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

/* ── Progress bar ───────────────────────────────────────────────── */
.stProgress > div > div { background: #00D4FF !important; }

/* ── Expander ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 10px !important;
}

/* ── Selectbox / slider ─────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #111827 !important;
    border-color: #1E3A5F !important;
    color: #F1F5F9 !important;
}
.stSlider [data-testid="stThumbValue"] { color: #00D4FF !important; }

/* ── Image comparison ───────────────────────────────────────────── */
.before-after-label {
    text-align:center;
    font-size:0.75rem;
    font-weight:700;
    letter-spacing:2px;
    text-transform:uppercase;
    padding:4px 0 8px;
}
.label-before { color:#64748B; }
.label-after  { color:#00D4FF; }

/* ── Detection item ─────────────────────────────────────────────── */
.det-item {
    display:flex;
    align-items:center;
    gap:10px;
    padding:7px 12px;
    background:#0A0E1A;
    border-radius:8px;
    margin:4px 0;
    border-left:3px solid;
}
.det-class { font-weight:700; font-size:0.9rem; min-width:110px; }
.det-conf  { font-size:0.8rem; color:#64748B; }
.det-src   { font-size:0.72rem; padding:2px 7px; border-radius:4px;
             background:#1E3A5F; color:#94A3B8; }

/* ── Scrollable detection list ─────────────────────────────────── */
.det-scroll {
    max-height: 340px;
    overflow-y: auto;
    padding-right: 4px;
}
.det-scroll::-webkit-scrollbar { width:4px; }
.det-scroll::-webkit-scrollbar-track { background:#0A0E1A; }
.det-scroll::-webkit-scrollbar-thumb { background:#1E3A5F; border-radius:4px; }

/* ── Header banner ──────────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1B2A 50%, #0A0E1A 100%);
    border: 1px solid #1E3A5F;
    border-top: 3px solid #00D4FF;
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:1rem;
}
.header-title { font-size:1.5rem; font-weight:800; color:#00D4FF; letter-spacing:1px; }
.header-sub   { font-size:0.8rem; color:#64748B; margin-top:3px; letter-spacing:.5px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE IMPORT — graceful fallback if modules not installed
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        from pipeline.ingest       import load_frame, validate_frame
        from pipeline.calibrate    import run_calibration, load_calibration
        from pipeline.deterministic import (run_deterministic,
                                             KalmanFrameFilter, BackgroundModel)
        from pipeline.enhance      import enhance_frame, detect_image_type
        from pipeline.detect       import (UnifiedDetector, draw_detections,
                                           detections_to_json, COCO_CLASSES)
        import yaml
        with open(ROOT / "params.yaml") as f:
            params = yaml.safe_load(f)

        calib    = load_calibration(params.get("paths", {})
                                    .get("calibration_dir", "data/calibration_assets"))
        detector = UnifiedDetector(
            yolo_model_size=params.get("detection", {}).get("yolo_model_size", "n"),
            yolo_conf      =params.get("detection", {}).get("yolo_conf", 0.25),
            use_thermal    =True,
            device         =params.get("detection", {}).get("device", "cpu"),
        )
        return {
            "ok": True,
            "load_frame":       load_frame,
            "validate_frame":   validate_frame,
            "run_calibration":  run_calibration,
            "run_deterministic":run_deterministic,
            "enhance_frame":    enhance_frame,
            "detect_image_type":detect_image_type,
            "draw_detections":  draw_detections,
            "detections_to_json":detections_to_json,
            "KalmanFrameFilter":KalmanFrameFilter,
            "BackgroundModel":  BackgroundModel,
            "calib":            calib,
            "detector":         detector,
            "params":           params,
            "coco_classes":     COCO_CLASSES,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# CORE: process one image → returns dict
# ══════════════════════════════════════════════════════════════════════════════
def process_image(img_array: np.ndarray, pl: dict,
                  conf_override: float = None,
                  do_sr: bool = False) -> dict:
    """Run full pipeline on a numpy array. Returns result dict."""

    if conf_override is not None:
        pl["detector"].yolo.conf = conf_override

    t0 = time.perf_counter()

    # Stage 1: Calibration
    calibrated = pl["run_calibration"](img_array, calib=pl["calib"])

    # Stage 2: Deterministic (stateless per-image — no cross-frame Kalman here)
    kalman   = pl["KalmanFrameFilter"](shape=calibrated.shape[:2])
    bg_model = pl["BackgroundModel"](alpha=0.05)
    processed = pl["run_deterministic"](
        calibrated, kalman=kalman, bg_model=bg_model,
        params=pl["params"].get("deterministic", {}))

    # Stage 3+4: Enhancement
    meta     = pl["detect_image_type"](img_array)
    enhanced = pl["enhance_frame"](processed, do_sr=do_sr, return_8bit=True)

    # Stage 5: Detection
    detections = pl["detector"].detect(enhanced, is_thermal=meta["is_thermal"])
    det_json   = pl["detections_to_json"](detections)

    # Annotated output
    annotated = pl["draw_detections"](enhanced, detections)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "original":    img_array,
        "enhanced":    enhanced,
        "annotated":   annotated,
        "detections":  det_json,
        "elapsed_ms":  elapsed_ms,
        "meta":        meta,
    }


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def pil_to_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.astype(np.float32)


def img_to_bytes(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    pil = np_to_pil(arr)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


# ── Detection class → badge colour ───────────────────────────────────────────
CLASS_COLOURS = {
    "person":"badge-green", "car":"badge-yell", "truck":"badge-yell",
    "bus":"badge-yell",     "motorcycle":"badge-cyan","bicycle":"badge-cyan",
    "airplane":"badge-blue","train":"badge-blue","boat":"badge-blue",
    "hotspot":"badge-red",  "fire":"badge-red", "heat_source":"badge-red",
    "anomaly":"badge-purp", "cold_region":"badge-blue",
    "dog":"badge-green",    "cat":"badge-green","bird":"badge-green",
}
BORDER_COLOURS = {
    "badge-green":"#10B981","badge-yell":"#F59E0B","badge-cyan":"#00D4FF",
    "badge-red":"#EF4444",  "badge-blue":"#3B82F6","badge-purp":"#8B5CF6",
}
def cls_badge(name: str) -> str:
    bc = CLASS_COLOURS.get(name, "badge-cyan")
    return f'<span class="badge {bc}">{name}</span>'

def det_border(name: str) -> str:
    bc = CLASS_COLOURS.get(name, "badge-cyan")
    return BORDER_COLOURS.get(bc, "#00D4FF")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(pl: dict) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 0 16px;">
          <div style="font-size:1.1rem;font-weight:800;color:#00D4FF;letter-spacing:1px;">
            🌡️ IR PIPELINE
          </div>
          <div style="font-size:0.72rem;color:#334155;letter-spacing:1.5px;
                      text-transform:uppercase;margin-top:2px;">
            DRDO SSPL · v2.0
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748B;'
                    'letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">'
                    'DETECTION SETTINGS</p>', unsafe_allow_html=True)

        conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                          help="Lower → more detections. Higher → stricter.")
        iou  = st.slider("NMS IoU threshold",    0.10, 0.80, 0.45, 0.05,
                          help="Controls overlap removal between boxes.")

        model_size = st.selectbox(
            "YOLOv8 model size",
            ["n (nano — fastest)", "s (small)", "m (medium)"],
            index=0,
            help="Larger = more accurate but slower"
        ).split()[0]

        st.markdown("---")
        st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748B;'
                    'letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">'
                    'ENHANCEMENT</p>', unsafe_allow_html=True)

        do_sr = st.toggle("Super-resolution ×2", value=False,
                           help="Requires EDSR_x2.pb in models/. Much slower.")
        show_raw = st.toggle("Show raw frame alongside", value=True)

        st.markdown("---")
        st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748B;'
                    'letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">'
                    'OUTPUT FORMAT</p>', unsafe_allow_html=True)
        out_fmt = st.radio("Download format", ["PNG", "JPEG"], horizontal=True)

        st.markdown("---")

        # Pipeline status
        if pl["ok"]:
            st.markdown("""
            <div class="ir-card ir-card-accent-green" style="padding:10px 12px;">
              <div style="font-size:0.72rem;font-weight:700;color:#10B981;
                          letter-spacing:1.5px;text-transform:uppercase;">
                ✓ PIPELINE READY
              </div>
              <div style="font-size:0.75rem;color:#64748B;margin-top:4px;">
                YOLO loaded · 85 classes
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Pipeline error:\n{pl.get('error','unknown')}")
            st.markdown("Run: `pip install -r requirements.txt`")

        st.markdown("---")
        st.markdown('<p style="font-size:0.7rem;color:#334155;text-align:center;">'
                    'ML + DL + MLOps · Self-hosted</p>', unsafe_allow_html=True)

    return {"conf": conf, "iou": iou, "model_size": model_size,
            "do_sr": do_sr, "show_raw": show_raw, "out_fmt": out_fmt}


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DISPLAY COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def render_before_after(result: dict, settings: dict, fname: str = "image"):
    """Show before/after comparison + metrics + detections."""
    original   = result["original"]
    annotated  = result["annotated"]
    detections = result["detections"]
    elapsed    = result["elapsed_ms"]

    # ── Before / After images ─────────────────────────────────────────────────
    if settings["show_raw"]:
        col_orig, col_enh = st.columns(2, gap="medium")
        with col_orig:
            st.markdown('<p class="before-after-label label-before">RAW INPUT</p>',
                        unsafe_allow_html=True)
            st.image(np_to_pil(original.astype(np.float32)
                     if original.dtype != np.uint8 else original),
                     use_container_width=True)
        with col_enh:
            st.markdown('<p class="before-after-label label-after">ENHANCED + DETECTED</p>',
                        unsafe_allow_html=True)
            st.image(np_to_pil(annotated), use_container_width=True)
    else:
        st.markdown('<p class="before-after-label label-after">ENHANCED + DETECTED</p>',
                    unsafe_allow_html=True)
        st.image(np_to_pil(annotated), use_container_width=True)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    n_det   = detections["count"]
    classes = list({d["label"] for d in detections["detections"]})
    yolo_c  = sum(1 for d in detections["detections"] if d["source"] == "yolo")
    therm_c = sum(1 for d in detections["detections"] if d["source"] == "thermal")

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="val" style="color:#00D4FF;">{n_det}</div>
        <div class="lbl">Objects Found</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#10B981;">{len(classes)}</div>
        <div class="lbl">Unique Classes</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#8B5CF6;">{yolo_c}</div>
        <div class="lbl">YOLO Detections</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#EF4444;">{therm_c}</div>
        <div class="lbl">Thermal Detections</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#F59E0B;">{elapsed:.0f}<span style="font-size:1rem;">ms</span></div>
        <div class="lbl">Process Time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detection list + download ─────────────────────────────────────────────
    dcol, djcol = st.columns([3, 2], gap="medium")

    with dcol:
        st.markdown('<div class="ir-card ir-card-accent-cyan">', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.75rem;font-weight:700;color:#00D4FF;'
                    f'letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">'
                    f'DETECTED OBJECTS — {n_det} total</p>', unsafe_allow_html=True)

        if n_det == 0:
            st.markdown('<p style="color:#334155;font-size:0.85rem;">'
                        'No objects detected. Try lowering the confidence threshold.</p>',
                        unsafe_allow_html=True)
        else:
            items_html = '<div class="det-scroll">'
            for d in sorted(detections["detections"],
                            key=lambda x: x["confidence"], reverse=True):
                border = det_border(d["label"])
                conf_pct = int(d["confidence"] * 100)
                src_txt  = "YOLO" if d["source"] == "yolo" else "THERMAL"
                items_html += f"""
                <div class="det-item" style="border-left-color:{border};">
                  <div class="det-class" style="color:{border};">{d['label']}</div>
                  <div style="flex:1;">
                    <div style="background:#1E3A5F;border-radius:3px;height:4px;width:100%;">
                      <div style="background:{border};height:4px;border-radius:3px;
                                  width:{conf_pct}%;"></div>
                    </div>
                  </div>
                  <div class="det-conf">{conf_pct}%</div>
                  <div class="det-src">{src_txt}</div>
                </div>"""
            items_html += '</div>'
            st.markdown(items_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with djcol:
        st.markdown('<div class="ir-card ir-card-accent-purp">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#8B5CF6;'
                    'letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">'
                    'DOWNLOAD RESULTS</p>', unsafe_allow_html=True)

        # Download enhanced image
        fmt      = settings.get("out_fmt", "PNG")
        img_bytes = img_to_bytes(annotated, fmt=fmt)
        st.download_button(
            label=f"⬇ Download Enhanced Image ({fmt})",
            data=img_bytes,
            file_name=f"{Path(fname).stem}_enhanced.{fmt.lower()}",
            mime=f"image/{fmt.lower()}",
            use_container_width=True,
        )

        # Download JSON detections
        json_bytes = json.dumps(detections, indent=2).encode()
        st.download_button(
            label="⬇ Download Detections (JSON)",
            data=json_bytes,
            file_name=f"{Path(fname).stem}_detections.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Raw JSON expander
        with st.expander("View raw detection JSON"):
            st.json(detections)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
def tab_single(pl: dict, settings: dict):
    st.markdown("""
    <div class="ir-card ir-card-accent-cyan">
      <p style="font-size:0.8rem;color:#94A3B8;margin:0;">
        Upload any image — thermal TIFF, PNG, JPEG, 16-bit RAW.
        The pipeline auto-detects image type and applies appropriate processing.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "npy"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;
                    border:1.5px dashed #1E3A5F;border-radius:12px;
                    background:#0A0E1A;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:12px;">🌡️</div>
          <div style="font-size:1rem;font-weight:600;color:#334155;margin-bottom:6px;">
            No image uploaded yet
          </div>
          <div style="font-size:0.8rem;color:#1E3A5F;">
            Supports: PNG · JPEG · TIFF (16-bit) · BMP · NPY
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not pl["ok"]:
        st.error("Pipeline not loaded. Check requirements.txt installation.")
        return

    # Load image
    try:
        if uploaded.name.endswith(".npy"):
            frame = np.load(io.BytesIO(uploaded.read())).astype(np.float32)
        else:
            pil   = Image.open(uploaded).convert("L")    # grayscale
            frame = np.array(pil).astype(np.float32)
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return

    valid, reason = pl["validate_frame"](frame)
    if not valid:
        st.error(f"Invalid image: {reason}")
        return

    # Info bar
    h, w = frame.shape[:2]
    st.markdown(f"""
    <div style="display:flex;gap:8px;align-items:center;margin:10px 0 16px;flex-wrap:wrap;">
      <span class="badge badge-cyan">📄 {uploaded.name}</span>
      <span class="badge badge-blue">{w}×{h}px</span>
      <span class="badge badge-purp">{frame.dtype}</span>
      <span class="badge badge-green">range {frame.min():.0f}–{frame.max():.0f}</span>
    </div>
    """, unsafe_allow_html=True)

    # Process button
    col_btn, col_sp = st.columns([2, 5])
    with col_btn:
        run = st.button("▶  Run Pipeline", use_container_width=True)

    if run:
        with st.spinner("Running 5-stage pipeline..."):
            try:
                result = process_image(
                    frame, pl,
                    conf_override=settings["conf"],
                    do_sr=settings["do_sr"],
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        st.success(f"Done in {result['elapsed_ms']:.0f} ms")
        st.markdown("---")
        render_before_after(result, settings, fname=uploaded.name)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def tab_batch(pl: dict, settings: dict):
    st.markdown("""
    <div class="ir-card ir-card-accent-yell">
      <p style="font-size:0.8rem;color:#94A3B8;margin:0;">
        Upload multiple images at once. All are processed through the full pipeline.
        Download all results as a ZIP when complete.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop images here or click to browse — select as many as you want",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "npy"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;
                    border:1.5px dashed #1E3A5F;border-radius:12px;
                    background:#0A0E1A;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:12px;">📁</div>
          <div style="font-size:1rem;font-weight:600;color:#334155;margin-bottom:6px;">
            No images uploaded yet
          </div>
          <div style="font-size:0.8rem;color:#1E3A5F;">
            Select multiple files with Ctrl+Click / Cmd+Click
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not pl["ok"]:
        st.error("Pipeline not loaded.")
        return

    # File summary
    total = len(uploaded_files)
    st.markdown(f"""
    <div style="display:flex;gap:8px;align-items:center;margin:10px 0 16px;flex-wrap:wrap;">
      <span class="badge badge-cyan">📁 {total} files queued</span>
      {"".join(f'<span class="badge badge-blue">{f.name[:24]}</span>'
               for f in uploaded_files[:6])}
      {'<span class="badge badge-purp">…and more</span>' if total > 6 else ''}
    </div>
    """, unsafe_allow_html=True)

    # Options row
    ocol1, ocol2, ocol3 = st.columns(3)
    with ocol1:
        show_each = st.toggle("Show each result inline", value=(total <= 10),
                               help="Disable for large batches to save memory.")
    with ocol2:
        stop_on_error = st.toggle("Stop on first error", value=False)
    with ocol3:
        run_batch = st.button("▶  Process All Images", use_container_width=True)

    if not run_batch:
        return

    # ── RUN BATCH ────────────────────────────────────────────────────────────
    progress_bar   = st.progress(0, text="Starting...")
    status_text    = st.empty()
    results_container = st.container()

    batch_results  = []   # (fname, result | None, error | None)
    zip_buffer     = io.BytesIO()
    fmt            = settings.get("out_fmt", "PNG")

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, uf in enumerate(uploaded_files):
            pct  = (i + 1) / total
            progress_bar.progress(pct,
                text=f"Processing {i+1}/{total}: {uf.name}")
            status_text.markdown(
                f'<span style="color:#64748B;font-size:0.85rem;">'
                f'⚡ {uf.name}</span>', unsafe_allow_html=True)

            try:
                # Load
                if uf.name.endswith(".npy"):
                    frame = np.load(io.BytesIO(uf.read())).astype(np.float32)
                else:
                    pil   = Image.open(uf).convert("L")
                    frame = np.array(pil).astype(np.float32)

                valid, reason = pl["validate_frame"](frame)
                if not valid:
                    raise ValueError(f"Invalid: {reason}")

                result = process_image(frame, pl,
                                       conf_override=settings["conf"],
                                       do_sr=settings["do_sr"])
                batch_results.append((uf.name, result, None))

                # Add to ZIP
                stem  = Path(uf.name).stem
                img_b = img_to_bytes(result["annotated"], fmt=fmt)
                zf.writestr(f"enhanced/{stem}_enhanced.{fmt.lower()}", img_b)
                det_b = json.dumps(result["detections"], indent=2).encode()
                zf.writestr(f"detections/{stem}_detections.json", det_b)

                # Inline preview
                if show_each:
                    with results_container:
                        with st.expander(
                            f"✅ {uf.name}  —  "
                            f"{result['detections']['count']} objects  |  "
                            f"{result['elapsed_ms']:.0f} ms",
                            expanded=(total <= 5)
                        ):
                            render_before_after(result, settings, fname=uf.name)

            except Exception as e:
                batch_results.append((uf.name, None, str(e)))
                if show_each:
                    with results_container:
                        st.error(f"❌ {uf.name}: {e}")
                if stop_on_error:
                    break

    progress_bar.progress(1.0, text="Complete!")
    status_text.empty()

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count  = sum(1 for _, r, e in batch_results if r is not None)
    err_count = sum(1 for _, r, e in batch_results if e is not None)
    total_det = sum(r["detections"]["count"]
                    for _, r, e in batch_results if r is not None)

    st.markdown(f"""
    <div class="metric-row" style="margin-top:1.5rem;">
      <div class="metric-tile">
        <div class="val" style="color:#00D4FF;">{ok_count}</div>
        <div class="lbl">Processed</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#EF4444;">{err_count}</div>
        <div class="lbl">Errors</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#10B981;">{total_det}</div>
        <div class="lbl">Total Objects</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#F59E0B;">
          {sum(r['elapsed_ms'] for _,r,e in batch_results if r)/max(ok_count,1):.0f}
          <span style="font-size:1rem;">ms</span>
        </div>
        <div class="lbl">Avg Per Image</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ZIP Download ──────────────────────────────────────────────────────────
    st.markdown('<div class="ir-card ir-card-accent-green" style="margin-top:1rem;">',
                unsafe_allow_html=True)
    st.download_button(
        label=f"⬇  Download All Results ({ok_count} images + JSONs) — ZIP",
        data=zip_buffer.getvalue(),
        file_name="ir_pipeline_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.markdown("ZIP contains: `enhanced/` folder (images) + `detections/` folder (JSONs)",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Error list
    if err_count > 0:
        with st.expander(f"⚠ {err_count} errors"):
            for fname, _, err in batch_results:
                if err:
                    st.markdown(f"- **{fname}**: `{err}`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT / PIPELINE INFO
# ══════════════════════════════════════════════════════════════════════════════
def tab_about():
    st.markdown("""
    <div class="ir-card ir-card-accent-cyan">
      <p style="font-size:1rem;font-weight:700;color:#00D4FF;margin-bottom:6px;">
        AI-Enhanced Cooled Sensor IR Imaging Pipeline
      </p>
      <p style="font-size:0.85rem;color:#94A3B8;margin:0;">
        5-stage automated image intelligence system. Fully self-hosted.
        Works on any image — thermal, grayscale, RGB. No cloud dependency.
      </p>
    </div>
    """, unsafe_allow_html=True)

    stages = [
        ("01", "#00D4FF", "Sensor Calibration",
         "Dark current · Flat field · Gain/offset · Temperature mapping · Linearity LUT",
         "Auto-estimates calibration from frame statistics if no pre-computed files exist"),
        ("02", "#3B82F6", "Deterministic Processing",
         "Bad pixel correction · NUC · Background subtraction · Kalman filtering · Wavelet denoising",
         "All algorithms are parameter-free. Self-tuning threshold estimation via MAD estimator"),
        ("03", "#8B5CF6", "ML Pre-processing",
         "Non-local means denoising · CLAHE contrast · Adaptive gamma · Unsharp masking",
         "Auto-detects noise level and adjusts denoising strength — works on any image type"),
        ("04", "#8B5CF6", "DL Enhancement",
         "Inpainting dead pixels · Optional EDSR super-resolution ×2",
         "EDSR uses pre-trained COCO weights — no fine-tuning required"),
        ("05", "#F59E0B", "Object Detection & Tracking",
         "YOLOv8-nano (80 COCO classes) + Thermal hotspot detector (5 IR classes)",
         "85 total detectable classes. Centroid tracker assigns persistent IDs across frames"),
    ]

    for num, color, title, algos, note in stages:
        st.markdown(f"""
        <div class="ir-card" style="border-left:3px solid {color};border-radius:10px;
                                    padding:12px 16px;margin-bottom:8px;">
          <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="font-size:1.2rem;font-weight:800;color:{color};
                        min-width:28px;font-family:monospace;">{num}</div>
            <div>
              <div style="font-size:0.95rem;font-weight:700;color:#F1F5F9;
                          margin-bottom:4px;">{title}</div>
              <div style="font-size:0.8rem;color:#64748B;margin-bottom:5px;">{algos}</div>
              <div style="font-size:0.78rem;color:{color};opacity:.8;">✦ {note}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="ir-card ir-card-accent-green">
          <p style="font-size:0.75rem;font-weight:700;color:#10B981;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
            DETECTABLE CLASSES
          </p>
          <p style="font-size:0.8rem;color:#94A3B8;line-height:1.7;">
            person · car · truck · bus · motorcycle · bicycle · airplane ·
            train · boat · traffic light · fire hydrant · stop sign · cat ·
            dog · bird · horse · cow · elephant · bear · zebra · giraffe ·
            <em>+59 more COCO classes</em><br><br>
            <span style="color:#EF4444;">Thermal-specific:</span>
            hotspot · fire · heat_source · anomaly · cold_region
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ir-card ir-card-accent-purp">
          <p style="font-size:0.75rem;font-weight:700;color:#8B5CF6;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
            ACCEPTED INPUT FORMATS
          </p>
          <p style="font-size:0.8rem;color:#94A3B8;line-height:1.9;">
            🌡 <strong>.tiff / .tif</strong> — 16-bit thermal sensor output<br>
            🔢 <strong>.npy</strong> — NumPy float32 arrays<br>
            🖼 <strong>.png</strong> — 8/16-bit lossless<br>
            📷 <strong>.jpg / .jpeg</strong> — standard photos<br>
            🗂 <strong>.bmp</strong> — uncompressed bitmap<br>
            ⚙️ <strong>.raw / .bin</strong> — binary sensor dumps (via API)
          </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Load pipeline (cached)
    with st.spinner("Loading pipeline..."):
        pl = load_pipeline()

    # Sidebar settings
    settings = render_sidebar(pl)

    # Header
    st.markdown("""
    <div class="header-banner">
      <div>
        <div class="header-title">🌡️ AI-ENHANCED IR IMAGING PIPELINE</div>
        <div class="header-sub">
          DRDO SSPL · ML + DL + MLOps · 85 Object Classes · Any Image Type
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
        <span class="badge badge-cyan">YOLOv8</span>
        <span class="badge badge-purp">U-Net</span>
        <span class="badge badge-green">COCO 80+</span>
        <span class="badge badge-yell">Wavelet</span>
        <span class="badge badge-red">Thermal</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "   🖼  Single Image   ",
        "   📁  Batch Upload   ",
        "   ℹ️  Pipeline Info   ",
    ])

    with tab1:
        tab_single(pl, settings)

    with tab2:
        tab_batch(pl, settings)

    with tab3:
        tab_about()


if __name__ == "__main__":
    main()
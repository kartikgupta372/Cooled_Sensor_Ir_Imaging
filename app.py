"""
app.py — Streamlit frontend for IR Imaging Pipeline
Run: streamlit run app.py
"""

import sys ,os ,io ,json ,time ,zipfile 
from pathlib import Path 

import streamlit as st 
import numpy as np 
import cv2 
from PIL import Image 

ROOT =Path (__file__ ).parent 
sys .path .insert (0 ,str (ROOT /"src"))

st .set_page_config (
page_title ="IR Imaging Pipeline",
page_icon ="🌡️",
layout ="wide",
initial_sidebar_state ="expanded",
)


st .markdown ("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #07090F !important; color: #F1F5F9 !important;
}
[data-testid="stSidebar"] {
    background-color: #0A0E1A !important;
    border-right: 1px solid #1A2A40 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
h1,h2,h3,h4 { color:#F1F5F9 !important; }
p, li, label, span { color:#CBD5E1 !important; }

.card {
    background:#0F1623; border:1px solid #1A2A40;
    border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:1rem;
}
.card-cyan  { border-top:3px solid #00D4FF; }
.card-green { border-top:3px solid #22C55E; }
.card-yell  { border-top:3px solid #F59E0B; }

.metric-row { display:flex; gap:10px; flex-wrap:wrap; margin:0.7rem 0; }
.metric-tile {
    background:#0F1623; border:1px solid #1A2A40; border-radius:10px;
    padding:12px 16px; min-width:100px; text-align:center; flex:1;
}
.metric-tile .val { font-size:1.55rem; font-weight:800; line-height:1.1; }
.metric-tile .lbl { font-size:0.68rem; letter-spacing:1.5px;
                    text-transform:uppercase; color:#475569; }

.badge {
    display:inline-block; font-size:0.72rem; font-weight:600;
    padding:3px 9px; border-radius:20px; margin:2px 3px; letter-spacing:.3px;
}
.bc { background:#00D4FF18;color:#00D4FF;border:1px solid #00D4FF33; }
.bg { background:#22C55E18;color:#22C55E;border:1px solid #22C55E33; }
.by { background:#F59E0B18;color:#F59E0B;border:1px solid #F59E0B33; }
.br { background:#EF444418;color:#EF4444;border:1px solid #EF444433; }

.stButton>button {
    background:#0A1628 !important; color:#00D4FF !important;
    border:1px solid #00D4FF44 !important; border-radius:8px !important;
    font-weight:600 !important;
}
.stButton>button:hover {
    background:#00D4FF !important; color:#07090F !important;
    border-color:#00D4FF !important;
}
[data-testid="stFileUploader"] {
    background:#0F1623 !important; border:1.5px dashed #1A2A40 !important;
    border-radius:12px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background:#0A0E1A !important; border-bottom:1px solid #1A2A40 !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#475569 !important;
    border-radius:8px 8px 0 0 !important; padding:8px 20px !important;
    font-weight:600 !important; font-size:0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background:#0F1623 !important; color:#00D4FF !important;
    border-bottom:2px solid #00D4FF !important;
}
.stProgress > div > div { background:#00D4FF !important; }
[data-testid="stExpander"] {
    background:#0F1623 !important; border:1px solid #1A2A40 !important;
    border-radius:10px !important;
}
.lbl-before { text-align:center; font-size:0.72rem; font-weight:700;
              letter-spacing:2px; text-transform:uppercase;
              color:#475569; padding:4px 0 6px; }
.lbl-after  { text-align:center; font-size:0.72rem; font-weight:700;
              letter-spacing:2px; text-transform:uppercase;
              color:#00D4FF; padding:4px 0 6px; }
.det-row {
    display:flex; align-items:center; gap:8px;
    padding:7px 10px; background:#07090F;
    border-radius:7px; margin:3px 0; border-left:3px solid;
}
.header {
    background:linear-gradient(135deg,#0A0E1A,#0D1B2A,#0A0E1A);
    border:1px solid #1A2A40; border-top:3px solid #00D4FF;
    border-radius:14px; padding:1.2rem 1.8rem;
    display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap;
    gap:1rem; margin-bottom:1.4rem;
}
.htitle { font-size:1.45rem; font-weight:800; color:#00D4FF; letter-spacing:1px; }
.hsub   { font-size:0.78rem; color:#475569; margin-top:3px; letter-spacing:.5px; }
</style>
""",unsafe_allow_html =True )





@st .cache_resource (show_spinner =False )
def load_pipeline ():
    try :
        from pipeline .ingest import validate_frame 
        from pipeline .calibrate import run_calibration ,load_calibration 
        from pipeline .deterministic import (run_deterministic ,
        KalmanFrameFilter ,BackgroundModel )
        from pipeline .enhance import enhance_frame ,detect_image_type 
        from pipeline .detect import (UnifiedDetector ,draw_detections ,
        detections_to_json )
        import yaml 
        with open (ROOT /"params.yaml")as f :
            params =yaml .safe_load (f )

        calib_dir =params .get ("paths",{}).get ("calibration_dir",
        "data/calibration_assets")
        calib =load_calibration (calib_dir )
        detector =UnifiedDetector (
        yolo_model_size =params .get ("detection",{}).get ("yolo_model_size","n"),
        yolo_conf =params .get ("detection",{}).get ("yolo_conf",0.40 ),
        use_thermal =False ,
        device =params .get ("detection",{}).get ("device","cpu"),
        )
        return dict (ok =True ,validate_frame =validate_frame ,
        run_calibration =run_calibration ,calib =calib ,
        run_deterministic =run_deterministic ,
        KalmanFrameFilter =KalmanFrameFilter ,
        BackgroundModel =BackgroundModel ,
        enhance_frame =enhance_frame ,
        detect_image_type =detect_image_type ,
        draw_detections =draw_detections ,
        detections_to_json =detections_to_json ,
        detector =detector ,params =params )
    except Exception as e :
        import traceback 
        return dict (ok =False ,error =str (e ),trace =traceback .format_exc ())





def run_pipeline (frame :np .ndarray ,pl :dict ,
conf :float ,do_sr :bool ,
use_thermal :bool )->dict :
    t0 =time .perf_counter ()


    pl ["detector"].conf =conf 
    pl ["detector"].use_thermal =use_thermal 


    calibrated =pl ["run_calibration"](frame ,calib =pl ["calib"])


    kal =pl ["KalmanFrameFilter"](shape =calibrated .shape [:2 ])


    processed =pl ["run_deterministic"](
    calibrated ,kalman =kal ,bg_model =None ,
    params =pl ["params"].get ("deterministic",{}))


    enhanced =pl ["enhance_frame"](processed ,do_sr =do_sr ,return_8bit =True ,params =pl ["params"].get ("enhance",{}))


    meta =pl ["detect_image_type"](frame )
    dets =pl ["detector"].detect (enhanced ,is_thermal =meta ["is_thermal"])
    det_json =pl ["detections_to_json"](dets )


    annotated =pl ["draw_detections"](enhanced ,dets )
    elapsed =(time .perf_counter ()-t0 )*1000 

    return dict (original =frame ,enhanced =enhanced ,
    annotated =annotated ,detections =det_json ,
    elapsed_ms =elapsed ,meta =meta )


def to_pil (arr :np .ndarray )->Image .Image :
    if arr .dtype !=np .uint8 :
        arr =cv2 .normalize (arr ,None ,0 ,255 ,cv2 .NORM_MINMAX ).astype (np .uint8 )
    if arr .ndim ==2 :
        return Image .fromarray (arr )
    return Image .fromarray (cv2 .cvtColor (arr ,cv2 .COLOR_BGR2RGB ))


def to_bytes (arr :np .ndarray ,fmt ="PNG")->bytes :
    buf =io .BytesIO ()
    to_pil (arr ).save (buf ,format =fmt )
    return buf .getvalue ()


def load_uploaded (uf )->np .ndarray :
    """Load any uploaded file → float32 2D grayscale array.
    Forces grayscale so the deterministic pipeline always gets (H,W) input.
    Handles: RGB, RGBA, L, P, 16-bit TIFF, NPY.
    """
    if uf .name .endswith (".npy"):
        arr =np .load (io .BytesIO (uf .read ()))
        if arr .ndim ==3 :

            arr =arr .mean (axis =2 )
        return arr .astype (np .float32 )
    pil =Image .open (uf )

    pil =pil .convert ("L")
    return np .array (pil ).astype (np .float32 )


DET_COLORS ={
"Person":"#22C55E",
"Vehicle":"#00D4FF",
"Motorcycle":"#00D4FF",
"Bicycle":"#A3E635",
"Aircraft":"#F59E0B",
"Watercraft":"#FB923C",
"Animal":"#C084FC",
"Screen":"#818CF8",
"Hotspot":"#EF4444",
}
def dcolor (label ):return DET_COLORS .get (label ,"#94A3B8")





def show_result (result :dict ,settings :dict ,fname :str ="image"):
    dets =result ["detections"]
    n =dets ["count"]
    elapsed =result ["elapsed_ms"]
    fmt =settings ["fmt"]


    if settings ["show_raw"]:
        c1 ,c2 =st .columns (2 ,gap ="medium")
        with c1 :
            st .markdown ('<p class="lbl-before">RAW INPUT</p>',
            unsafe_allow_html =True )
            st .image (to_pil (result ["original"]),use_container_width =True )
        with c2 :
            st .markdown ('<p class="lbl-after">ENHANCED + DETECTED</p>',
            unsafe_allow_html =True )
            st .image (to_pil (result ["annotated"]),use_container_width =True )
    else :
        st .markdown ('<p class="lbl-after">ENHANCED + DETECTED</p>',
        unsafe_allow_html =True )
        st .image (to_pil (result ["annotated"]),use_container_width =True )


    classes =list ({d ["label"]for d in dets ["detections"]})
    st .markdown (f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="val" style="color:#00D4FF;">{n }</div>
        <div class="lbl">Objects</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#22C55E;">{len (classes )}</div>
        <div class="lbl">Classes</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#F59E0B;">{elapsed :.0f}<span style="font-size:.9rem">ms</span></div>
        <div class="lbl">Process Time</div>
      </div>
    </div>
    """,unsafe_allow_html =True )


    dl ,dr =st .columns ([3 ,2 ],gap ="medium")

    with dl :
        st .markdown ('<div class="card card-cyan">',unsafe_allow_html =True )
        st .markdown (f'<p style="font-size:.73rem;font-weight:700;color:#00D4FF;'
        f'letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">'
        f'DETECTIONS — {n } objects</p>',unsafe_allow_html =True )

        if n ==0 :
            st .markdown ('<p style="color:#334155;font-size:.85rem;">No objects found. '
            'Try lowering the confidence threshold in the sidebar.</p>',
            unsafe_allow_html =True )
        else :
            items =""
            for d in sorted (dets ["detections"],
            key =lambda x :x ["confidence"],reverse =True ):
                c =dcolor (d ["label"])
                pct =int (d ["confidence"]*100 )
                items +=f"""
                <div class="det-row" style="border-left-color:{c };">
                  <div style="font-weight:700;color:{c };min-width:100px;
                              font-size:.88rem;">{d ['label']}</div>
                  <div style="flex:1;">
                    <div style="background:#1A2A40;border-radius:3px;height:4px;">
                      <div style="background:{c };height:4px;border-radius:3px;
                                  width:{pct }%;"></div></div></div>
                  <div style="font-size:.78rem;color:#475569;">{pct }%</div>
                  <div style="font-size:.7rem;background:#1A2A40;padding:2px 6px;
                              border-radius:4px;color:#94A3B8;">{d ['source'].upper ()}</div>
                </div>"""
            st .markdown (
            f'<div style="max-height:300px;overflow-y:auto;">{items }</div>',
            unsafe_allow_html =True )
        st .markdown ("</div>",unsafe_allow_html =True )

    with dr :
        st .markdown ('<div class="card card-yell">',unsafe_allow_html =True )
        st .markdown ('<p style="font-size:.73rem;font-weight:700;color:#F59E0B;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">'
        'DOWNLOADS</p>',unsafe_allow_html =True )

        stem =Path (fname ).stem 
        st .download_button (
        f"⬇  Enhanced image ({fmt })",
        data =to_bytes (result ["annotated"],fmt ),
        file_name =f"{stem }_enhanced.{fmt .lower ()}",
        mime =f"image/{fmt .lower ()}",
        use_container_width =True ,
        )
        st .download_button (
        "⬇  Detections JSON",
        data =json .dumps (dets ,indent =2 ).encode (),
        file_name =f"{stem }_detections.json",
        mime ="application/json",
        use_container_width =True ,
        )
        st .markdown ("</div>",unsafe_allow_html =True )

        with st .expander ("Raw JSON"):
            st .json (dets )





def sidebar (pl ):
    with st .sidebar :
        st .markdown ("""
        <div style="padding:10px 0 14px;">
          <div style="font-size:1.05rem;font-weight:800;color:#00D4FF;letter-spacing:1px;">
            🌡️ IR PIPELINE
          </div>
          <div style="font-size:.7rem;color:#334155;letter-spacing:1.5px;
                      text-transform:uppercase;margin-top:2px;">
            DRDO SSPL · v2.0
          </div>
        </div>""",unsafe_allow_html =True )

        st .markdown ("---")
        st .markdown ('<p style="font-size:.72rem;font-weight:700;color:#475569;'
        'letter-spacing:2px;text-transform:uppercase;">DETECTION</p>',
        unsafe_allow_html =True )

        conf =st .slider ("Confidence",0.10 ,0.90 ,0.40 ,0.05 ,
        help ="Raise to reduce false positives. Lower to catch more.")
        use_thermal =st .toggle ("Thermal hotspot detector",value =False ,
        help ="Adds large-area hotspot detection. Leave OFF for normal photos.")
        show_raw =st .toggle ("Show raw frame",value =True )

        st .markdown ("---")
        st .markdown ('<p style="font-size:.72rem;font-weight:700;color:#475569;'
        'letter-spacing:2px;text-transform:uppercase;">ENHANCEMENT</p>',
        unsafe_allow_html =True )
        do_sr =st .toggle ("Super-resolution ×2",value =False ,
        help ="Needs models/EDSR_x2.pb. Much slower.")

        st .markdown ("---")
        fmt =st .radio ("Download format",["PNG","JPEG"],horizontal =True )

        st .markdown ("---")
        if pl ["ok"]:
            st .markdown ("""
            <div class="card card-green" style="padding:9px 11px;">
              <div style="font-size:.7rem;font-weight:700;color:#22C55E;
                          letter-spacing:1.5px;text-transform:uppercase;">
                ✓ PIPELINE READY
              </div>
              <div style="font-size:.73rem;color:#475569;margin-top:3px;">
                YOLOv8 · 8 display classes
              </div>
            </div>""",unsafe_allow_html =True )
        else :
            st .error (f"Pipeline error — see console")
            with st .expander ("Error details"):
                st .code (pl .get ("trace",pl .get ("error","")))

    return dict (conf =conf ,use_thermal =use_thermal ,
    show_raw =show_raw ,do_sr =do_sr ,fmt =fmt )





def tab_single (pl ,s ):
    st .markdown ('<div class="card card-cyan"><p style="font-size:.82rem;'
    'color:#94A3B8;margin:0;">Upload any image — PNG, JPEG, 16-bit TIFF, NPY. '
    'Pipeline auto-adapts to image type.</p></div>',
    unsafe_allow_html =True )

    uf =st .file_uploader (
    "Drop image here",
    type =["png","jpg","jpeg","tiff","tif","bmp","npy"],
    label_visibility ="collapsed",
    )

    if uf is None :
        st .markdown ("""
        <div style="text-align:center;padding:3.5rem 1rem;
                    border:1.5px dashed #1A2A40;border-radius:12px;
                    background:#0A0E1A;margin-top:1rem;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🌡️</div>
          <div style="font-size:.95rem;font-weight:600;color:#334155;">
            No image uploaded</div>
          <div style="font-size:.78rem;color:#1A2A40;margin-top:4px;">
            PNG · JPEG · TIFF · BMP · NPY</div>
        </div>""",unsafe_allow_html =True )
        return 

    if not pl ["ok"]:
        st .error ("Pipeline not loaded. Run: pip install -r requirements.txt")
        return 

    try :
        frame =load_uploaded (uf )
    except Exception as e :
        st .error (f"Could not load image: {e }")
        return 

    valid ,reason =pl ["validate_frame"](frame )
    if not valid :
        st .error (f"Invalid image: {reason }")
        return 

    h ,w =frame .shape [:2 ]
    st .markdown (f"""
    <div style="display:flex;gap:6px;align-items:center;margin:8px 0 14px;flex-wrap:wrap;">
      <span class="badge bc">📄 {uf .name }</span>
      <span class="badge bc">{w }×{h }</span>
      <span class="badge bc">{frame .dtype }</span>
      <span class="badge bg">range {frame .min ():.0f}–{frame .max ():.0f}</span>
    </div>""",unsafe_allow_html =True )

    cb ,_ =st .columns ([2 ,5 ])
    with cb :
        run =st .button ("▶  Run Pipeline",use_container_width =True )

    if run :
        with st .spinner ("Enhancing + detecting…"):
            try :
                result =run_pipeline (frame ,pl ,
                conf =s ["conf"],do_sr =s ["do_sr"],
                use_thermal =s ["use_thermal"])
            except Exception as e :
                import traceback 
                st .error (f"Pipeline error: {e }")
                st .code (traceback .format_exc ())
                return 
        st .success (f"✓ Done in {result ['elapsed_ms']:.0f} ms")
        st .markdown ("---")
        show_result (result ,s ,fname =uf .name )


def tab_batch (pl ,s ):
    st .markdown ('<div class="card card-yell"><p style="font-size:.82rem;'
    'color:#94A3B8;margin:0;">Upload multiple images. '
    'All processed through the full pipeline. '
    'Download all as ZIP when complete.</p></div>',
    unsafe_allow_html =True )

    ufs =st .file_uploader (
    "Drop images here — select many with Ctrl+Click",
    type =["png","jpg","jpeg","tiff","tif","bmp","npy"],
    accept_multiple_files =True ,
    label_visibility ="collapsed",
    )

    if not ufs :
        st .markdown ("""
        <div style="text-align:center;padding:3.5rem 1rem;
                    border:1.5px dashed #1A2A40;border-radius:12px;
                    background:#0A0E1A;margin-top:1rem;">
          <div style="font-size:2.5rem;margin-bottom:10px;">📁</div>
          <div style="font-size:.95rem;font-weight:600;color:#334155;">
            No files uploaded</div>
          <div style="font-size:.78rem;color:#1A2A40;margin-top:4px;">
            Ctrl+Click to select multiple files</div>
        </div>""",unsafe_allow_html =True )
        return 

    if not pl ["ok"]:
        st .error ("Pipeline not loaded.")
        return 

    total =len (ufs )
    st .markdown (f'<span class="badge bc">📁 {total } files queued</span>'
    +"".join (f'<span class="badge bc">{f .name [:20 ]}</span>'
    for f in ufs [:5 ])
    +('<span class="badge by">+more</span>'if total >5 else ""),
    unsafe_allow_html =True )

    oc1 ,oc2 ,oc3 =st .columns (3 )
    with oc1 :
        show_each =st .toggle ("Show each result",value =(total <=8 ))
    with oc2 :
        stop_err =st .toggle ("Stop on error",value =False )
    with oc3 :
        go =st .button ("▶  Process All",use_container_width =True )

    if not go :
        return 

    prog =st .progress (0 ,text ="Starting…")
    stat =st .empty ()
    box =st .container ()
    batch =[]
    zip_buf =io .BytesIO ()
    fmt =s ["fmt"]

    with zipfile .ZipFile (zip_buf ,"w",zipfile .ZIP_DEFLATED )as zf :
        for i ,uf in enumerate (ufs ):
            prog .progress ((i +1 )/total ,
            text =f"Processing {i +1 }/{total }: {uf .name }")
            stat .markdown (f'<span style="font-size:.82rem;color:#475569;">'
            f'⚡ {uf .name }</span>',unsafe_allow_html =True )
            try :
                frame =load_uploaded (uf )
                valid ,reason =pl ["validate_frame"](frame )
                if not valid :
                    raise ValueError (reason )

                result =run_pipeline (frame ,pl ,conf =s ["conf"],
                do_sr =s ["do_sr"],
                use_thermal =s ["use_thermal"])
                batch .append ((uf .name ,result ,None ))

                stem =Path (uf .name ).stem 
                zf .writestr (f"enhanced/{stem }_enhanced.{fmt .lower ()}",
                to_bytes (result ["annotated"],fmt ))
                zf .writestr (f"detections/{stem }.json",
                json .dumps (result ["detections"],indent =2 ))

                if show_each :
                    with box :
                        with st .expander (
                        f"✅ {uf .name } — "
                        f"{result ['detections']['count']} objects — "
                        f"{result ['elapsed_ms']:.0f}ms",
                        expanded =(total <=4 )):
                            show_result (result ,s ,fname =uf .name )
            except Exception as e :
                batch .append ((uf .name ,None ,str (e )))
                if show_each :
                    with box :
                        st .error (f"❌ {uf .name }: {e }")
                if stop_err :
                    break 

    prog .progress (1.0 ,text ="Complete!")
    stat .empty ()

    ok_n =sum (1 for _ ,r ,_ in batch if r )
    err_n =sum (1 for _ ,_ ,e in batch if e )
    tot_d =sum (r ["detections"]["count"]for _ ,r ,_ in batch if r )
    avg_t =(sum (r ["elapsed_ms"]for _ ,r ,_ in batch if r )
    /max (ok_n ,1 ))

    st .markdown (f"""
    <div class="metric-row" style="margin-top:1.2rem;">
      <div class="metric-tile">
        <div class="val" style="color:#00D4FF;">{ok_n }</div>
        <div class="lbl">Done</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#EF4444;">{err_n }</div>
        <div class="lbl">Errors</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#22C55E;">{tot_d }</div>
        <div class="lbl">Objects Found</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:#F59E0B;">{avg_t :.0f}<span style="font-size:.85rem">ms</span></div>
        <div class="lbl">Avg / Image</div>
      </div>
    </div>""",unsafe_allow_html =True )

    st .markdown ('<div class="card card-green" style="margin-top:.8rem;">',
    unsafe_allow_html =True )
    st .download_button (
    f"⬇  Download all ({ok_n } enhanced images + JSONs) — ZIP",
    data =zip_buf .getvalue (),
    file_name ="ir_pipeline_results.zip",
    mime ="application/zip",
    use_container_width =True ,
    )
    st .markdown ("</div>",unsafe_allow_html =True )

    if err_n :
        with st .expander (f"⚠ {err_n } errors"):
            for fname ,_ ,err in batch :
                if err :
                    st .markdown (f"- **{fname }**: `{err }`")


def tab_about ():
    st .markdown ("""
    <div class="card card-cyan">
      <p style="font-size:.95rem;font-weight:700;color:#00D4FF;margin-bottom:5px;">
        AI-Enhanced Cooled Sensor IR Imaging Pipeline — v2.0
      </p>
      <p style="font-size:.82rem;color:#94A3B8;margin:0;">
        5-stage pipeline. DRDO SSPL. Self-hosted. Any image type.
      </p>
    </div>""",unsafe_allow_html =True )

    for num ,col ,title ,detail in [
    ("01","#00D4FF","Calibration","Dark current · Flat field · Gain/offset · Auto-estimates if files missing"),
    ("02","#3B82F6","Deterministic","Bad pixel fix · NUC · Kalman filter · Wavelet denoise"),
    ("03","#8B5CF6","Enhancement","Bilateral denoise → Gamma → CLAHE → Unsharp mask (1.8×) → Laplacian boost"),
    ("04","#8B5CF6","DL / SR","Dead pixel inpaint · Optional EDSR ×2 super-resolution"),
    ("05","#F59E0B","Detection","YOLOv8n COCO → 8 display groups · Centroid tracker · Optional thermal"),
    ]:
        st .markdown (f"""
        <div style="background:#0F1623;border:1px solid #1A2A40;border-left:3px solid {col };
                    border-radius:10px;padding:11px 15px;margin-bottom:7px;display:flex;gap:10px;">
          <div style="font-size:1.1rem;font-weight:800;color:{col };
                      font-family:monospace;min-width:24px;">{num }</div>
          <div>
            <div style="font-size:.9rem;font-weight:700;color:#F1F5F9;">{title }</div>
            <div style="font-size:.78rem;color:#475569;margin-top:3px;">{detail }</div>
          </div>
        </div>""",unsafe_allow_html =True )

    st .markdown ("---")
    c1 ,c2 =st .columns (2 )
    with c1 :
        st .markdown ("""
        <div class="card card-green">
          <p style="font-size:.73rem;font-weight:700;color:#22C55E;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            DISPLAY CLASSES (8)
          </p>
          <p style="font-size:.82rem;color:#94A3B8;line-height:1.9;">
            👤 Person &nbsp;🚗 Vehicle &nbsp;🏍 Motorcycle<br>
            🚲 Bicycle &nbsp;✈️ Aircraft &nbsp;🚢 Watercraft<br>
            🐕 Animal &nbsp;💻 Screen<br>
            <span style="color:#EF4444;">+ Hotspot</span> (thermal detector, opt-in)
          </p>
        </div>""",unsafe_allow_html =True )
    with c2 :
        st .markdown ("""
        <div class="card card-yell">
          <p style="font-size:.73rem;font-weight:700;color:#F59E0B;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            INPUT FORMATS
          </p>
          <p style="font-size:.82rem;color:#94A3B8;line-height:1.9;">
            🌡 TIFF / TIF — 16-bit thermal<br>
            🔢 NPY — NumPy float32<br>
            🖼 PNG — 8/16-bit lossless<br>
            📷 JPG / JPEG — standard<br>
            🗂 BMP — uncompressed
          </p>
        </div>""",unsafe_allow_html =True )





def main ():
    with st .spinner ("Loading pipeline…"):
        pl =load_pipeline ()

    s =sidebar (pl )

    st .markdown ("""
    <div class="header">
      <div>
        <div class="htitle">🌡️ AI-ENHANCED IR IMAGING PIPELINE</div>
        <div class="hsub">DRDO SSPL · ML + DL + MLOps · Self-hosted</div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
        <span class="badge bc">YOLOv8</span>
        <span class="badge bg">8 Clean Classes</span>
        <span class="badge by">Wavelet</span>
        <span class="badge br">CLAHE 3×</span>
      </div>
    </div>""",unsafe_allow_html =True )

    t1 ,t2 ,t3 =st .tabs ([
    "   🖼  Single Image   ",
    "   📁  Batch Upload   ",
    "   ℹ️  About   ",
    ])
    with t1 :tab_single (pl ,s )
    with t2 :tab_batch (pl ,s )
    with t3 :tab_about ()


if __name__ =="__main__":
    main ()

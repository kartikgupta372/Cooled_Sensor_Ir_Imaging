"""
ingest.py — Universal image ingestion.
Accepts ANY file format, ANY bit depth, single file OR entire folder.
Auto-detects format, validates, and normalises to a consistent internal format.
"""

import os 
import cv2 
import numpy as np 
from pathlib import Path 
from typing import List ,Tuple ,Generator 

SUPPORTED_EXTENSIONS ={
".tiff",".tif",
".npy",
".raw",".bin",
".png",".jpg",".jpeg",
".bmp",".webp",
}


def load_frame (path :str ,
sensor_shape :Tuple [int ,int ]=(480 ,640 ))->np .ndarray :
    """
    Load any single image file → float32 ndarray.
    sensor_shape used only for .raw/.bin files that have no header.
    Returns float32 (H, W) for grayscale or (H, W, 3) for colour.
    """
    path =str (path )
    ext =Path (path ).suffix .lower ()

    if not os .path .exists (path ):
        raise FileNotFoundError (f"File not found: {path }")


    if ext ==".npy":
        frame =np .load (path )
        return _coerce_dtype (frame )


    if ext in (".raw",".bin"):
        raw =np .fromfile (path ,dtype =np .uint16 )
        h ,w =sensor_shape 
        if raw .size >=h *w :
            frame =raw [:h *w ].reshape ((h ,w )).astype (np .float32 )
        else :

            side =int (np .sqrt (raw .size ))
            frame =raw [:side *side ].reshape ((side ,side )).astype (np .float32 )
        return frame 


    if ext in (".tiff",".tif"):
        try :
            import tifffile 
            frame =tifffile .imread (path )
            return _coerce_dtype (frame )
        except ImportError :
            pass 


    frame =cv2 .imread (path ,cv2 .IMREAD_UNCHANGED )
    if frame is None :
        raise ValueError (f"Could not read image: {path }")

    return _coerce_dtype (frame )


def _coerce_dtype (frame :np .ndarray )->np .ndarray :
    """
    Normalise any array to float32.
    Preserves values — does NOT range-normalise to [0,1] here
    (that happens in enhance.py adaptive_normalize).
    """
    if frame .dtype ==np .uint8 :
        return frame .astype (np .float32 )
    if frame .dtype ==np .uint16 :
        return frame .astype (np .float32 )
    if frame .dtype in (np .float32 ,np .float64 ):
        return frame .astype (np .float32 )
    return frame .astype (np .float32 )



def discover_folder (folder_path :str )->List [str ]:
    """
    Recursively find all supported image files in a folder.
    Returns sorted list of absolute paths.
    """
    folder =Path (folder_path )
    if not folder .exists ():
        raise FileNotFoundError (f"Folder not found: {folder_path }")

    found =[]
    for ext in SUPPORTED_EXTENSIONS :
        found .extend (folder .rglob (f"*{ext }"))
        found .extend (folder .rglob (f"*{ext .upper ()}"))

    found =sorted (set (found ))
    if not found :
        raise ValueError (
        f"No supported images found in {folder_path }. "
        f"Supported: {SUPPORTED_EXTENSIONS }"
        )
    return [str (p )for p in found ]


def load_folder (folder_path :str ,
sensor_shape :Tuple [int ,int ]=(480 ,640 ),
max_frames :int =None )->Generator :
    """
    Generator: yields (path, frame) tuples from a folder.
    Use as: for path, frame in load_folder('/my/data'):
    max_frames: cap at N frames (useful for testing).
    """
    paths =discover_folder (folder_path )
    if max_frames :
        paths =paths [:max_frames ]

    for path in paths :
        try :
            frame =load_frame (path ,sensor_shape =sensor_shape )
            yield path ,frame 
        except Exception as e :
            print (f"[INGEST] Skipping {path }: {e }")
            continue 


def validate_frame (frame :np .ndarray )->Tuple [bool ,str ]:
    """
    Check a frame is usable.
    Returns (is_valid, reason_if_invalid).
    """
    if frame is None :
        return False ,"Frame is None"
    if frame .size ==0 :
        return False ,"Empty array"
    h =frame .shape [0 ]
    w =frame .shape [1 ]if frame .ndim >1 else 1 
    if h <32 or w <32 :
        return False ,f"Too small: {h }x{w }"
    if not np .isfinite (frame ).any ():
        return False ,"All values are NaN or Inf"
    return True ,"ok"


def get_frame_info (frame :np .ndarray ,path :str ="")->dict :
    """Return metadata dict about a loaded frame."""
    h =frame .shape [0 ]
    w =frame .shape [1 ]if frame .ndim >1 else 1 
    c =frame .shape [2 ]if frame .ndim ==3 else 1 
    return {
    "path":path ,
    "shape":frame .shape ,
    "height":h ,
    "width":w ,
    "channels":c ,
    "dtype":str (frame .dtype ),
    "min":float (np .nanmin (frame )),
    "max":float (np .nanmax (frame )),
    "mean":float (np .nanmean (frame )),
    "std":float (np .nanstd (frame )),
    "has_nan":bool (np .isnan (frame ).any ()),
    "has_inf":bool (np .isinf (frame ).any ()),
    }

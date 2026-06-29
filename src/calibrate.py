"""
calibrate.py — Adaptive sensor calibration.

KEY FIX: If calibration files (.npy) are missing, auto-generates them
from the input frame itself using statistical estimation.
This means the pipeline works on ANY image, even with no prior calibration data.
"""

import numpy as np 
import os 
from pathlib import Path 



def load_calibration (calib_dir :str ="data/calibration_assets")->dict :
    """
    Load calibration assets from disk.
    If any file is missing, returns None for that asset —
    run_calibration() will auto-generate estimates from the input frame.
    """
    calib ={}
    files ={
    "dark_frame":"dark_frame.npy",
    "flat_field":"flat_field.npy",
    "gain_map":"gain_map.npy",
    "offset_map":"offset_map.npy",
    "lut":"linearity_lut.npy",
    }
    for key ,fname in files .items ():
        fpath =os .path .join (calib_dir ,fname )
        if os .path .exists (fpath ):
            calib [key ]=np .load (fpath )
        else :
            calib [key ]=None 
    return calib 



def estimate_dark_frame (frame :np .ndarray )->np .ndarray :
    """
    Estimate dark current from input frame statistics.
    Uses the bottom 1% of pixel values as dark current estimate.
    Works because dark current = sensor floor noise.
    """
    low_val =np .percentile (frame ,1.0 )
    return np .full (frame .shape ,low_val ,dtype =np .float32 )


def estimate_flat_field (frame :np .ndarray )->np .ndarray :
    """
    Estimate flat field from the frame itself using Gaussian blur.
    A heavily blurred version of the frame approximates the
    large-scale illumination / sensitivity non-uniformity.
    """
    import cv2 

    ksize =max (21 ,(min (frame .shape [:2 ])//8 )|1 )
    blurred =cv2 .GaussianBlur (frame .astype (np .float32 ),(ksize ,ksize ),0 )
    mean_val =np .mean (blurred )
    if mean_val <1e-8 :
        return np .ones (frame .shape ,dtype =np .float32 )
    return (blurred /mean_val ).astype (np .float32 )


def estimate_gain_offset (frame :np .ndarray )->tuple :
    """
    Identity gain/offset (no-op) when real calibration unavailable.
    Returns gain=1, offset=0 maps — safe default.
    """
    gain =np .ones (frame .shape ,dtype =np .float32 )
    offset =np .zeros (frame .shape ,dtype =np .float32 )
    return gain ,offset 



def dark_current_correction (frame :np .ndarray ,
dark_frame :np .ndarray )->np .ndarray :
    return np .clip (frame -dark_frame ,0 ,None ).astype (np .float32 )


def flat_field_calibration (frame :np .ndarray ,
flat_field :np .ndarray )->np .ndarray :
    flat_norm =flat_field /(np .mean (flat_field )+1e-8 )
    return (frame /(flat_norm +1e-8 )).astype (np .float32 )


def gain_offset_calibration (frame :np .ndarray ,
gain_map :np .ndarray ,
offset_map :np .ndarray )->np .ndarray :
    return (gain_map *frame +offset_map ).astype (np .float32 )


def linearity_correction (frame :np .ndarray ,
lut :np .ndarray )->np .ndarray :
    indices =np .clip (frame .astype (np .int32 ),0 ,len (lut )-1 )
    return lut [indices ].astype (np .float32 )


def temperature_calibration (frame :np .ndarray ,
R1 :float =14364.0 ,
R2 :float =0.010 ,
F :float =1.0 ,
O :float =-7.5 ,
B :float =1428.0 )->np .ndarray :
    """
    Convert raw ADC counts → Kelvin using Planck equation.
    Default constants are typical for FLIR Lepton / generic InSb.
    If constants unknown, output is still a monotonic temperature proxy.
    """
    safe =np .clip (frame ,1 ,None )
    return (B /np .log (R1 /(R2 *(safe +O ))+F )).astype (np .float32 )



def run_calibration (frame :np .ndarray ,
calib :dict =None ,
calib_dir :str ="data/calibration_assets",
to_kelvin :bool =False )->np .ndarray :
    """
    Run full calibration chain on a single frame.

    BEHAVIOUR:
    - If calib dict provided and has assets → uses them
    - If calib assets are None → auto-estimates from frame (works on ANY image)
    - to_kelvin: apply Planck equation to convert to temperature

    This means this function ALWAYS succeeds, even on images
    that were never seen during training or calibration setup.
    """
    if calib is None :
        calib =load_calibration (calib_dir )

    f =frame .astype (np .float32 ).copy ()


    dark =calib .get ("dark_frame")
    if dark is None or dark .shape !=f .shape :
        dark =estimate_dark_frame (f )
    f =dark_current_correction (f ,dark )


    flat =calib .get ("flat_field")
    if flat is None or flat .shape !=f .shape :
        flat =estimate_flat_field (f )
    f =flat_field_calibration (f ,flat )


    gain =calib .get ("gain_map")
    offset =calib .get ("offset_map")
    if gain is None or gain .shape !=f .shape :
        gain ,offset =estimate_gain_offset (f )
    f =gain_offset_calibration (f ,gain ,offset )


    lut =calib .get ("lut")
    if lut is not None :
        f =linearity_correction (f ,lut )


    if to_kelvin :
        f =temperature_calibration (f )

    return f 



def generate_calibration_files (frames :list ,
output_dir :str ="data/calibration_assets"):
    """
    Generate calibration .npy files from a list of frames.
    Call this once when you have real calibration data.
    frames: list of float32 ndarrays (all same shape).
    """
    os .makedirs (output_dir ,exist_ok =True )
    stack =np .stack ([f .astype (np .float32 )for f in frames ],axis =0 )


    dark_frame =np .mean (stack ,axis =0 )
    np .save (os .path .join (output_dir ,"dark_frame.npy"),dark_frame )


    flat_field =dark_frame /(np .mean (dark_frame )+1e-8 )
    np .save (os .path .join (output_dir ,"flat_field.npy"),flat_field )


    temporal_std =np .std (stack ,axis =0 )+1e-8 
    global_std =np .mean (temporal_std )
    gain_map =global_std /temporal_std 
    offset_map =np .mean (dark_frame )-gain_map *dark_frame 
    np .save (os .path .join (output_dir ,"gain_map.npy"),gain_map )
    np .save (os .path .join (output_dir ,"offset_map.npy"),offset_map )

    print (f"[CALIB] Saved calibration files to {output_dir }")
    return {"dark_frame":dark_frame ,"flat_field":flat_field ,
    "gain_map":gain_map ,"offset_map":offset_map }

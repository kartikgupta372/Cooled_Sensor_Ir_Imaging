import os 
import yaml 
import numpy as np 
import cv2 
from tqdm import tqdm 

def load_params (yaml_path ="params.yaml"):
    """Loads our configuration from the central params.yaml file."""
    with open (yaml_path ,"r")as f :
        return yaml .safe_load (f )

def scale_to_8bit (image ):
    """
    CLAHE in OpenCV only works on 8-bit (0-255) or 16-bit images.
    Our 14-bit data sits inside a uint16 container but only uses
    values 0-16383. We normalize to 0-255 (uint8) for CLAHE processing.

    Formula:  scaled = (pixel / max_possible_value) * 255
    """
    max_val =16383.0 
    scaled =(image .astype (np .float32 )/max_val )*255.0 
    return scaled .astype (np .uint8 )

def scale_to_16bit (image ):
    """
    After CLAHE, we scale back from 8-bit to the 14-bit range inside
    a uint16 container, so all downstream scripts stay consistent.
    """
    max_val =16383.0 
    scaled =(image .astype (np .float32 )/255.0 )*max_val 
    return scaled .astype (np .uint16 )

def apply_clahe (image_8bit ,clip_limit ,grid_size ):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image_8bit  : uint8 numpy array (the 8-bit scaled IR image)
        clip_limit  : How aggressively to boost contrast (from params.yaml).
                      Higher = more contrast but also more noise amplification.
        grid_size   : Tuple (rows, cols) — the image is divided into this 
                      many tiles. Each tile gets its own histogram equalization.
    
    Returns:
        enhanced    : uint8 numpy array with improved local contrast
    """

    clahe =cv2 .createCLAHE (
    clipLimit =clip_limit ,
    tileGridSize =tuple (grid_size )
    )


    enhanced =clahe .apply (image_8bit )
    return enhanced 

def apply_denoise (image ,weight ):
    """
    Applies a light Gaussian blur to suppress any noise amplified by CLAHE.
    
    The weight controls blending between the noisy CLAHE output and the
    smoother blurred version:
        result = (1 - weight) * original + weight * blurred
    
    At weight=0.1, we keep 90% original detail and soften 10%.
    """
    blurred =cv2 .GaussianBlur (image ,ksize =(3 ,3 ),sigmaX =0 )
    blended =cv2 .addWeighted (image ,1.0 -weight ,blurred ,weight ,0 )
    return blended 

def process_image (image ,params ):
    """Applies the full deterministic processing chain to a single image."""
    clip_limit =params ["process"]["clahe_clip_limit"]
    grid_size =params ["process"]["clahe_grid_size"]
    denoise_w =params ["process"]["denoise_weight"]


    image_8bit =scale_to_8bit (image )


    enhanced_8bit =apply_clahe (image_8bit ,clip_limit ,grid_size )


    denoised_8bit =apply_denoise (enhanced_8bit ,denoise_w )


    result =scale_to_16bit (denoised_8bit )

    return result 

def main ():
    params =load_params ()
    in_dir =params ["process"]["input_dir"]
    out_dir =params ["process"]["output_dir"]

    os .makedirs (out_dir ,exist_ok =True )

    image_files =sorted ([f for f in os .listdir (in_dir )if f .endswith (".tiff")])

    if len (image_files )==0 :
        print ("No images found — run calibrate.py first!")
        return 

    print (f"Processing {len (image_files )} images with CLAHE...")

    for filename in tqdm (image_files ):
        in_path =os .path .join (in_dir ,filename )
        out_path =os .path .join (out_dir ,filename )


        image =cv2 .imread (in_path ,cv2 .IMREAD_UNCHANGED )

        processed =process_image (image ,params )

        cv2 .imwrite (out_path ,processed )

    print (f"Saved to {out_dir }")

if __name__ =="__main__":
    main ()

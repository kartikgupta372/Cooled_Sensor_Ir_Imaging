"""
deterministic.py — Adaptive deterministic processing.
All algorithms are parameter-free or self-tuning — works on ANY image.
"""

import cv2 
import numpy as np 
from scipy .ndimage import median_filter 
import pywt 



def detect_bad_pixels (frame :np .ndarray ,
sigma_threshold :float =5.0 )->np .ndarray :
    """Pixels > N sigma from 3x3 neighbourhood median are bad."""
    median =cv2 .medianBlur (frame .astype (np .float32 ),3 )
    diff =np .abs (frame -median )
    std =np .std (diff )
    return diff >(sigma_threshold *std )


def correct_bad_pixels (frame :np .ndarray ,
bad_mask :np .ndarray )->np .ndarray :
    corrected =frame .copy ()
    corrected [bad_mask ]=median_filter (frame ,size =3 )[bad_mask ]
    return corrected 



def nuc_scene_based (frame :np .ndarray ,
gain_map :np .ndarray =None ,
offset_map :np .ndarray =None )->np .ndarray :
    """
    If maps provided → apply them.
    If not → estimate from frame statistics (self-contained NUC).
    """
    if gain_map is not None and gain_map .shape ==frame .shape :
        return (gain_map *frame +(offset_map or 0 )).astype (np .float32 )


    h ,w =frame .shape [:2 ]
    result =frame .astype (np .float32 ).copy ()
    ts =64 
    global_mean =float (np .mean (frame ))

    for r in range (0 ,h ,ts ):
        for c in range (0 ,w ,ts ):
            tile =frame [r :r +ts ,c :c +ts ]
            tile_mean =float (np .mean (tile ))
            if tile_mean >1e-8 :
                result [r :r +ts ,c :c +ts ]=tile *(global_mean /tile_mean )

    return result 



class BackgroundModel :
    """Exponential moving average background model."""
    def __init__ (self ,alpha :float =0.05 ):
        self .alpha =alpha 
        self .bg =None 

    def update (self ,frame :np .ndarray )->np .ndarray :
        if self .bg is None :


            self .bg =frame .astype (np .float32 ).copy ()
            return frame .astype (np .float32 )
        self .bg =self .alpha *frame +(1 -self .alpha )*self .bg 


        result =frame .astype (np .float32 )-self .bg 

        if float (np .mean (result ))<-5.0 :
            return frame .astype (np .float32 )
        return np .clip (result ,0 ,None )



class KalmanFrameFilter :
    def __init__ (self ,shape ,process_noise =0.001 ,measurement_noise =0.1 ):
        self .x =None 
        self .P =None 
        self .Q =process_noise 
        self .R =measurement_noise 

    def update (self ,measurement :np .ndarray )->np .ndarray :
        if self .x is None or self .x .shape !=measurement .shape :
            self .x =np .zeros (measurement .shape ,dtype =np .float32 )
            self .P =np .ones (measurement .shape ,dtype =np .float32 )

        P_pred =self .P +self .Q 
        K =P_pred /(P_pred +self .R )
        self .x =self .x +K *(measurement .astype (np .float32 )-self .x )
        self .P =(1 -K )*P_pred 
        return self .x .copy ()



def wavelet_denoise (frame :np .ndarray ,
wavelet :str ="db4",
level :int =3 ,
mode :str ="soft")->np .ndarray :
    f =frame .astype (np .float32 )

    if f .ndim ==3 :
        f =f .mean (axis =2 )
    elif f .ndim !=2 :
        return frame .astype (np .float32 )

    lo ,hi =f .min (),f .max ()
    if hi -lo <1e-8 :
        return f 
    fn =(f -lo )/(hi -lo )

    coeffs =pywt .wavedec2 (fn ,wavelet =wavelet ,level =level )
    sigma =np .median (np .abs (coeffs [-1 ][0 ]))/0.6745 
    thr =sigma *np .sqrt (2 *np .log (fn .size +1 ))

    denoised =[coeffs [0 ]]+[
    tuple (pywt .threshold (c ,thr ,mode =mode )for c in detail )
    for detail in coeffs [1 :]
    ]
    result =pywt .waverec2 (denoised ,wavelet =wavelet )
    result =result [:frame .shape [0 ],:frame .shape [1 ]]

    return (np .clip (result ,0 ,1 )*(hi -lo )+lo ).astype (np .float32 )



def adaptive_histogram_equalization (frame :np .ndarray ,
clip_limit :float =2.0 ,
tile_size :int =8 )->np .ndarray :
    norm =cv2 .normalize (frame ,None ,0 ,255 ,cv2 .NORM_MINMAX ).astype (np .uint8 )
    clahe =cv2 .createCLAHE (clipLimit =clip_limit ,
    tileGridSize =(tile_size ,tile_size ))
    return clahe .apply (norm ).astype (np .float32 )



def wiener_filter (frame :np .ndarray ,kernel_size :int =5 )->np .ndarray :
    from scipy .signal import wiener 
    return wiener (frame .astype (np .float32 ),mysize =kernel_size ).astype (np .float32 )



def run_deterministic (frame :np .ndarray ,
kalman :KalmanFrameFilter =None ,
bg_model :BackgroundModel =None ,
params :dict =None )->np .ndarray :
    """
    Run full deterministic pipeline on one frame.
    Stateful components (Kalman, BG) are passed in so batch
    processing maintains state across frames.
    All steps are adaptive — work on any image without prior calibration.
    """
    p =params or {}
    f =frame .astype (np .float32 )


    if f .ndim ==3 :
        nc =f .shape [2 ]
        if nc ==4 :
            f =cv2 .cvtColor (f .clip (0 ,255 ).astype (np .uint8 ),
            cv2 .COLOR_BGRA2GRAY ).astype (np .float32 )
        elif nc ==3 :
            f =cv2 .cvtColor (f .clip (0 ,255 ).astype (np .uint8 ),
            cv2 .COLOR_BGR2GRAY ).astype (np .float32 )
        elif nc ==1 :
            f =f [:,:,0 ]
    elif f .ndim >3 :
        raise ValueError (f"run_deterministic: unsupported frame ndim={f .ndim }")


    bad =detect_bad_pixels (f ,p .get ("bad_pixel_threshold_sigma",5.0 ))
    f =correct_bad_pixels (f ,bad )





    if bg_model is not None :
        f =bg_model .update (f )

        f =np .clip (f ,0 ,None )


    if kalman is not None :
        f =kalman .update (f )


    f =wavelet_denoise (
    f ,
    wavelet =p .get ("wavelet","db4"),
    level =p .get ("wavelet_level",3 ),
    mode =p .get ("wavelet_threshold_mode","soft"),
    )

    return f 

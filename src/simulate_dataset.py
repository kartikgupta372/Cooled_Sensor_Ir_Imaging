import os 
import cv2 
import numpy as np 
import yaml 
from urllib import request 

def load_params (yaml_path ="params.yaml"):
    with open (yaml_path ,"r")as f :
        return yaml .safe_load (f )

def download_base_images (cache_dir ="data/cache"):
    os .makedirs (cache_dir ,exist_ok =True )
    urls =[
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    ]
    base_images =[]


    for i ,url in enumerate (urls ):
        path =os .path .join (cache_dir ,f"base_{i }.jpg")
        if not os .path .exists (path ):
            request .urlretrieve (url ,path )
        img =cv2 .imread (path ,cv2 .IMREAD_GRAYSCALE )
        base_images .append (cv2 .resize (img ,(640 ,480 )))


    try :
        import skimage .data 
        sk_images =[
        skimage .data .astronaut (),
        skimage .data .camera (),
        skimage .data .chelsea (),
        skimage .data .horse (),
        skimage .data .coffee (),
        ]
        for img in sk_images :
            if img .dtype ==bool :
                img =(img *255 ).astype (np .uint8 )
            if len (img .shape )==3 :
                img =cv2 .cvtColor (img ,cv2 .COLOR_RGB2GRAY )
            base_images .append (cv2 .resize (img ,(640 ,480 )))
    except ImportError :
        pass 

    return base_images 

def generate_simulated_data ():
    """Generates clean ground-truth IR targets and raw noisy IR data."""
    params =load_params ()
    num_images =params ['simulate']['num_images']
    image_size =(params ['base']['image_height'],params ['base']['image_width'])
    noise_level =params ['simulate']['dark_current_noise']


    os .makedirs ('data/raw',exist_ok =True )
    os .makedirs ('data/clean',exist_ok =True )

    print ("Downloading structural base images...")
    base_images =download_base_images ()

    print (f"Generating {num_images } simulated IR images based on real structures...")


    max_val =16383 

    for i in range (num_images ):

        base =base_images [i %len (base_images )]



        clean_14bit =(base .astype (np .float32 )/255.0 *max_val ).astype (np .uint16 )



        noise =np .random .normal (0 ,noise_level *max_val ,image_size )
        raw_float =clean_14bit .astype (np .float32 )+noise 


        y ,x =np .mgrid [0 :image_size [0 ],0 :image_size [1 ]]
        gradient =(x /image_size [1 ]+y /image_size [0 ])*(0.1 *max_val )
        raw_float +=gradient 


        num_bad_pixels =int (image_size [0 ]*image_size [1 ]*0.005 )
        bad_y =np .random .randint (0 ,image_size [0 ],num_bad_pixels )
        bad_x =np .random .randint (0 ,image_size [1 ],num_bad_pixels )
        raw_float [bad_y ,bad_x ]=np .random .choice ([0 ,max_val ],num_bad_pixels )


        raw_14bit =np .clip (raw_float ,0 ,max_val ).astype (np .uint16 )


        cv2 .imwrite (f'data/clean/sim_{i :04d}.tiff',clean_14bit )
        cv2 .imwrite (f'data/raw/sim_{i :04d}.tiff',raw_14bit )

    print (" Saved clean and raw images successfully!")

if __name__ =="__main__":
    generate_simulated_data ()

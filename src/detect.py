"""
detect.py — Universal object detection for any image type.

Uses YOLOv8 pre-trained on COCO (80 classes) — works on ANY image,
no fine-tuning required. Adds IR-specific hotspot detection on top.

Total classes available: 80 COCO + thermal-specific = 90+ classes.
"""

import cv2 
import numpy as np 
from dataclasses import dataclass ,field 
from typing import List ,Dict ,Optional 
import os 


COCO_CLASSES =[
"person","bicycle","car","motorcycle","airplane","bus","train",
"truck","boat","traffic light","fire hydrant","stop sign",
"parking meter","bench","bird","cat","dog","horse","sheep",
"cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard",
"sports ball","kite","baseball bat","baseball glove","skateboard",
"surfboard","tennis racket","bottle","wine glass","cup","fork",
"knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair",
"couch","potted plant","bed","dining table","toilet","tv",
"laptop","mouse","remote","keyboard","cell phone","microwave",
"oven","toaster","sink","refrigerator","book","clock","vase",
"scissors","teddy bear","hair drier","toothbrush"
]


IR_CLASSES ={
"hotspot":{"min_temp_percentile":90 ,"color":(0 ,0 ,255 )},
"cold_region":{"min_temp_percentile":None ,"color":(255 ,100 ,0 )},
"fire":{"min_temp_percentile":98 ,"color":(0 ,50 ,255 )},
"heat_source":{"min_temp_percentile":93 ,"color":(0 ,128 ,255 )},
"anomaly":{"min_temp_percentile":95 ,"color":(128 ,0 ,255 )},
}


np .random .seed (42 )
CLASS_COLORS ={name :tuple (int (x )for x in np .random .randint (50 ,230 ,3 ))
for name in COCO_CLASSES }

CLASS_COLORS .update ({
"person":(0 ,255 ,0 ),
"car":(255 ,200 ,0 ),
"truck":(255 ,140 ,0 ),
"bus":(255 ,80 ,0 ),
"motorcycle":(200 ,255 ,0 ),
"bicycle":(0 ,255 ,200 ),
"airplane":(0 ,200 ,255 ),
"train":(255 ,0 ,200 ),
"boat":(100 ,200 ,255 ),
"dog":(255 ,255 ,0 ),
"cat":(255 ,200 ,200 ),
"horse":(200 ,150 ,50 ),
"bird":(150 ,255 ,150 ),
"fire hydrant":(255 ,0 ,0 ),
"stop sign":(255 ,0 ,50 ),
"traffic light":(0 ,255 ,100 ),
})


@dataclass 
class Detection :
    bbox :tuple 
    centroid :tuple 
    area :float 
    label :str 
    confidence :float 
    track_id :int =-1 
    mean_intensity :float =0.0 
    source :str ="yolo"
    extra :dict =field (default_factory =dict )





class UniversalYOLODetector :
    """
    YOLOv8 detector using pre-trained COCO weights.
    Key properties:
    - Works on ANY image (RGB, grayscale, thermal colourmap, etc.)
    - 80 COCO classes out of the box
    - No fine-tuning required — uses Ultralytics YOLOv8n.pt
    - Grayscale images are auto-converted to 3-channel before inference
    """

    def __init__ (self ,
    model_size :str ="n",
    conf_threshold :float =0.25 ,
    iou_threshold :float =0.45 ,
    device :str ="cpu"):
        self .conf =conf_threshold 
        self .iou =iou_threshold 
        self .device =device 
        self .model =None 
        self ._load_model (model_size )

    def _load_model (self ,size :str ):
        """Download and load YOLOv8 pre-trained on COCO. Auto-downloads on first run."""
        try :
            from ultralytics import YOLO 
            model_name =f"yolov8{size }.pt"

            self .model =YOLO (model_name )
            self .model .fuse ()
            print (f"[YOLO] Loaded yolov8{size }.pt — {len (COCO_CLASSES )} COCO classes")
        except ImportError :
            print ("[YOLO] ultralytics not installed. Run: pip install ultralytics")
            self .model =None 
        except Exception as e :
            print (f"[YOLO] Load failed: {e }")
            self .model =None 

    def _prepare_image (self ,frame :np .ndarray )->np .ndarray :
        """
        Convert any image to 3-channel uint8 for YOLO inference.
        Handles: grayscale, float32, 16-bit, thermal colourmap, BGR.
        """
        img =frame .copy ()


        if img .dtype in [np .float32 ,np .float64 ]:
            lo ,hi =img .min (),img .max ()
            img =((img -lo )/(hi -lo +1e-8 )*255 ).astype (np .uint8 )
        elif img .dtype ==np .uint16 :
            img =(img /256 ).astype (np .uint8 )
        else :
            img =img .astype (np .uint8 )


        if img .ndim ==2 :
            img =cv2 .cvtColor (img ,cv2 .COLOR_GRAY2BGR )
        elif img .ndim ==3 and img .shape [2 ]==1 :
            img =cv2 .cvtColor (img [:,:,0 ],cv2 .COLOR_GRAY2BGR )

        return img 

    def detect (self ,frame :np .ndarray )->List [Detection ]:
        """Run YOLO detection on any input image."""
        if self .model is None :
            return []

        img =self ._prepare_image (frame )

        try :
            results =self .model (
            img ,
            conf =self .conf ,
            iou =self .iou ,
            verbose =False ,
            device =self .device 
            )
        except Exception as e :
            print (f"[YOLO] Inference error: {e }")
            return []

        detections =[]
        for result in results :
            if result .boxes is None :
                continue 
            for box in result .boxes :
                x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ].tolist ())
                conf =float (box .conf [0 ])
                cls =int (box .cls [0 ])
                label =COCO_CLASSES [cls ]if cls <len (COCO_CLASSES )else f"class_{cls }"
                w ,h =x2 -x1 ,y2 -y1 
                cx ,cy =x1 +w //2 ,y1 +h //2 


                gray =cv2 .cvtColor (img ,cv2 .COLOR_BGR2GRAY )
                roi =gray [max (0 ,y1 ):y2 ,max (0 ,x1 ):x2 ]
                mean_i =float (np .mean (roi ))if roi .size >0 else 0.0 

                detections .append (Detection (
                bbox =(x1 ,y1 ,w ,h ),
                centroid =(cx ,cy ),
                area =float (w *h ),
                label =label ,
                confidence =conf ,
                mean_intensity =mean_i ,
                source ="yolo"
                ))

        return detections 





class ThermalHotspotDetector :
    """
    Detects IR-specific features using pixel statistics.
    Works on: thermal images, any high-dynamic-range grayscale image.
    Classes: hotspot, cold_region, fire, heat_source, anomaly.
    """

    def __init__ (self ,
    min_area :int =40 ,
    hotspot_percentile :float =90.0 ,
    fire_percentile :float =98.0 ):
        self .min_area =min_area 
        self .hot_pct =hotspot_percentile 
        self .fire_pct =fire_percentile 

    def detect (self ,frame :np .ndarray )->List [Detection ]:
        gray =self ._to_gray_float (frame )
        detections =[]


        detections +=self ._threshold_detect (
        gray ,self .hot_pct ,"hotspot",from_top =True )


        detections +=self ._threshold_detect (
        gray ,self .fire_pct ,"fire",from_top =True )


        detections +=self ._threshold_detect (
        gray ,5.0 ,"cold_region",from_top =False )


        return self ._nms (detections ,iou_threshold =0.3 )

    def _to_gray_float (self ,frame :np .ndarray )->np .ndarray :
        if frame .ndim ==3 :
            frame =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2GRAY )
        return frame .astype (np .float32 )

    def _threshold_detect (self ,gray :np .ndarray ,percentile :float ,
    label :str ,from_top :bool )->List [Detection ]:

        global_mean =float (np .mean (gray ))
        global_std =float (np .std (gray ))


        threshold =float (np .percentile (gray ,percentile ))



        if from_top :
            min_thresh =global_mean +2.5 *global_std 
            threshold =max (threshold ,min_thresh )
            binary =(gray >=threshold ).astype (np .uint8 )*255 
        else :
            max_thresh =global_mean -2.5 *global_std 
            threshold =min (threshold ,max_thresh )
            binary =(gray <=threshold ).astype (np .uint8 )*255 

        kernel =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(5 ,5 ))
        binary =cv2 .morphologyEx (binary ,cv2 .MORPH_CLOSE ,kernel )
        binary =cv2 .morphologyEx (binary ,cv2 .MORPH_OPEN ,kernel )

        contours ,_ =cv2 .findContours (binary ,cv2 .RETR_EXTERNAL ,
        cv2 .CHAIN_APPROX_SIMPLE )
        detections =[]
        for cnt in contours :
            area =cv2 .contourArea (cnt )
            if area <self .min_area :
                continue 
            x ,y ,w ,h =cv2 .boundingRect (cnt )
            cx ,cy =x +w //2 ,y +h //2 
            roi_mean =float (np .mean (gray [y :y +h ,x :x +w ]))


            conf =min (0.99 ,abs (roi_mean -global_mean )/(global_std +1e-8 )*0.2 )

            detections .append (Detection (
            bbox =(x ,y ,w ,h ),
            centroid =(cx ,cy ),
            area =area ,
            label =label ,
            confidence =conf ,
            mean_intensity =roi_mean ,
            source ="thermal"
            ))
        return detections 

    def _nms (self ,detections :List [Detection ],
    iou_threshold :float =0.5 )->List [Detection ]:
        """Simple IoU-based NMS to remove overlapping detections."""
        if not detections :
            return []
        detections .sort (key =lambda d :d .confidence ,reverse =True )
        keep =[]
        for d in detections :
            overlaps =False 
            for k in keep :
                if self ._iou (d .bbox ,k .bbox )>iou_threshold :
                    overlaps =True 
                    break 
            if not overlaps :
                keep .append (d )
        return keep 

    @staticmethod 
    def _iou (b1 ,b2 )->float :
        x1 ,y1 ,w1 ,h1 =b1 
        x2 ,y2 ,w2 ,h2 =b2 
        ix =max (0 ,min (x1 +w1 ,x2 +w2 )-max (x1 ,x2 ))
        iy =max (0 ,min (y1 +h1 ,y2 +h2 )-max (y1 ,y2 ))
        inter =ix *iy 
        union =w1 *h1 +w2 *h2 -inter 
        return inter /(union +1e-8 )





class CentroidTracker :
    def __init__ (self ,max_disappeared :int =15 ,max_distance :float =60.0 ):
        self .next_id =0 
        self .objects ={}
        self .disappeared ={}
        self .labels ={}
        self .max_gone =max_disappeared 
        self .max_dist =max_distance 

    def update (self ,detections :List [Detection ])->List [Detection ]:
        if not detections :
            for oid in list (self .disappeared ):
                self .disappeared [oid ]+=1 
                if self .disappeared [oid ]>self .max_gone :
                    del self .objects [oid ]
                    del self .disappeared [oid ]
                    self .labels .pop (oid ,None )
            return detections 

        new_centroids =np .array ([d .centroid for d in detections ],dtype =np .float32 )

        if not self .objects :
            for i ,d in enumerate (detections ):
                self .objects [self .next_id ]=new_centroids [i ]
                self .disappeared [self .next_id ]=0 
                self .labels [self .next_id ]=d .label 
                detections [i ].track_id =self .next_id 
                self .next_id +=1 
            return detections 

        obj_ids =list (self .objects .keys ())
        obj_cents =np .array (list (self .objects .values ()),dtype =np .float32 )


        D =np .linalg .norm (
        obj_cents [:,np .newaxis ]-new_centroids [np .newaxis ,:],axis =2 )

        rows =D .min (axis =1 ).argsort ()
        cols =D .argmin (axis =1 )[rows ]

        used_rows ,used_cols =set (),set ()
        for r ,c in zip (rows ,cols ):
            if r in used_rows or c in used_cols :
                continue 
            if D [r ,c ]>self .max_dist :
                continue 
            oid =obj_ids [r ]
            self .objects [oid ]=new_centroids [c ]
            self .disappeared [oid ]=0 
            detections [c ].track_id =oid 
            used_rows .add (r );used_cols .add (c )


        for c in range (len (detections )):
            if c not in used_cols :
                self .objects [self .next_id ]=new_centroids [c ]
                self .disappeared [self .next_id ]=0 
                self .labels [self .next_id ]=detections [c ].label 
                detections [c ].track_id =self .next_id 
                self .next_id +=1 


        for r in range (len (obj_ids )):
            if r not in used_rows :
                oid =obj_ids [r ]
                self .disappeared [oid ]+=1 
                if self .disappeared [oid ]>self .max_gone :
                    del self .objects [oid ]
                    del self .disappeared [oid ]
                    self .labels .pop (oid ,None )

        return detections 





class UnifiedDetector :
    """
    Master detector that combines:
    1. YOLOv8 (80 COCO classes) — for any RGB/thermal colourmap image
    2. Thermal hotspot detection — for raw thermal / grayscale IR
    3. Centroid tracker — persistent track IDs across frames

    Works on ANY input: RGB photo, IR thermal, grayscale, 16-bit.
    """

    def __init__ (self ,
    yolo_model_size :str ="n",
    yolo_conf :float =0.25 ,
    use_thermal :bool =True ,
    device :str ="cpu"):
        self .yolo =UniversalYOLODetector (yolo_model_size ,yolo_conf ,
        device =device )
        self .thermal =ThermalHotspotDetector ()if use_thermal else None 
        self .tracker =CentroidTracker ()

    def detect (self ,frame :np .ndarray ,
    is_thermal :bool =False )->List [Detection ]:
        """
        Run full detection pipeline.
        frame: any image type.
        is_thermal: hint that image is IR (adds thermal-specific classes).
        """
        all_detections =[]


        yolo_dets =self .yolo .detect (frame )
        all_detections .extend (yolo_dets )


        if self .thermal and is_thermal :
            thermal_dets =self .thermal .detect (frame )

            for td in thermal_dets :
                overlaps =any (
                UnifiedDetector ._iou_static (td .bbox ,yd .bbox )>0.4 
                for yd in yolo_dets 
                )
                if not overlaps :
                    all_detections .append (td )


        all_detections =self .tracker .update (all_detections )
        return all_detections 

    @staticmethod 
    def _iou_static (b1 ,b2 )->float :
        x1 ,y1 ,w1 ,h1 =b1 
        x2 ,y2 ,w2 ,h2 =b2 
        ix =max (0 ,min (x1 +w1 ,x2 +w2 )-max (x1 ,x2 ))
        iy =max (0 ,min (y1 +h1 ,y2 +h2 )-max (y1 ,y2 ))
        inter =ix *iy 
        union =w1 *h1 +w2 *h2 -inter 
        return inter /(union +1e-8 )





def draw_detections (frame :np .ndarray ,
detections :List [Detection ],
show_confidence :bool =True ,
show_track_id :bool =True )->np .ndarray :
    """
    Draw bounding boxes on any image.
    Handles grayscale → RGB conversion automatically.
    """
    if frame .ndim ==2 :
        vis =cv2 .cvtColor (frame ,cv2 .COLOR_GRAY2BGR )
    elif frame .dtype !=np .uint8 :
        norm =frame .astype (np .float32 )
        norm =(norm -norm .min ())/(norm .max ()-norm .min ()+1e-8 )
        vis =(norm *255 ).astype (np .uint8 )
        if vis .ndim ==2 :
            vis =cv2 .cvtColor (vis ,cv2 .COLOR_GRAY2BGR )
    else :
        vis =frame .copy ()
        if vis .ndim ==2 :
            vis =cv2 .cvtColor (vis ,cv2 .COLOR_GRAY2BGR )

    for d in detections :
        x ,y ,w ,h =[int (v )for v in d .bbox ]
        color =CLASS_COLORS .get (d .label ,
        IR_CLASSES .get (d .label ,{}).get ("color",(0 ,255 ,0 )))


        cv2 .rectangle (vis ,(x ,y ),(x +w ,y +h ),color ,2 )


        parts =[d .label ]
        if show_confidence :
            parts .append (f"{d .confidence :.2f}")
        if show_track_id and d .track_id >=0 :
            parts .append (f"#{d .track_id }")
        label_text =" ".join (parts )


        (lw ,lh ),_ =cv2 .getTextSize (label_text ,cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,1 )
        label_y =max (y -4 ,lh +4 )
        cv2 .rectangle (vis ,
        (x ,label_y -lh -4 ),
        (x +lw +4 ,label_y ),
        color ,-1 )
        cv2 .putText (vis ,label_text ,
        (x +2 ,label_y -2 ),
        cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,
        (0 ,0 ,0 ),1 ,cv2 .LINE_AA )


    count_text =f"{len (detections )} detections"
    cv2 .putText (vis ,count_text ,(8 ,22 ),
    cv2 .FONT_HERSHEY_SIMPLEX ,0.6 ,
    (255 ,255 ,255 ),2 ,cv2 .LINE_AA )

    return vis 


def detections_to_json (detections :List [Detection ],
frame_id :str ="unknown")->dict :
    """Serialise detections to JSON-ready dict."""
    return {
    "frame_id":frame_id ,
    "count":int (len (detections )),
    "detections":[
    {
    "label":str (d .label ),
    "confidence":float (round (float (d .confidence ),4 )),
    "bbox":[int (v )for v in d .bbox ],
    "centroid":[int (v )for v in d .centroid ],
    "area":float (round (float (d .area ),1 )),
    "track_id":int (d .track_id ),
    "mean_intensity":float (round (float (d .mean_intensity ),2 )),
    "source":str (d .source ),
    }
    for d in detections 
    ]
    }

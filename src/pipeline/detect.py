"""
detect.py — Clean, minimal object detection.

Philosophy (fixed from v1):
- YOLO only. Thermal hotspot detector is OFF by default — it was
  generating 70+ noise detections on every image.
- 80 COCO classes collapsed into 8 clean display groups:
  Person · Vehicle · Animal · Bicycle · Aircraft · Watercraft · Electronics · Other
- Labels show ONLY the group name — no confidence score, no #track_id clutter.
- High confidence threshold (0.40) + minimum area filter removes tiny noise boxes.
- Result looks like Image 3: 3–6 clean boxes on a readable image.
"""

import cv2 
import numpy as np 
from dataclasses import dataclass ,field 
from typing import List ,Optional 


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
"scissors","teddy bear","hair drier","toothbrush",
]


CLASS_GROUP ={

"person":("Person",(0 ,220 ,80 )),

"car":("Vehicle",(0 ,200 ,255 )),
"truck":("Vehicle",(0 ,200 ,255 )),
"bus":("Vehicle",(0 ,200 ,255 )),
"train":("Vehicle",(0 ,200 ,255 )),
"motorcycle":("Motorcycle",(0 ,200 ,255 )),

"bicycle":("Bicycle",(100 ,255 ,180 )),

"airplane":("Aircraft",(255 ,200 ,0 )),

"boat":("Watercraft",(255 ,150 ,0 )),

"bird":("Animal",(200 ,100 ,255 )),
"cat":("Animal",(200 ,100 ,255 )),
"dog":("Animal",(200 ,100 ,255 )),
"horse":("Animal",(200 ,100 ,255 )),
"sheep":("Animal",(200 ,100 ,255 )),
"cow":("Animal",(200 ,100 ,255 )),
"elephant":("Animal",(200 ,100 ,255 )),
"bear":("Animal",(200 ,100 ,255 )),
"zebra":("Animal",(200 ,100 ,255 )),
"giraffe":("Animal",(200 ,100 ,255 )),

"tv":("Screen",(180 ,180 ,255 )),
"laptop":("Screen",(180 ,180 ,255 )),
"cell phone":("Screen",(180 ,180 ,255 )),

}


IGNORED_CLASSES ={
"fork","knife","spoon","bowl","banana","apple","sandwich",
"orange","broccoli","carrot","hot dog","pizza","donut","cake",
"chair","couch","potted plant","bed","dining table","toilet",
"bottle","wine glass","cup","backpack","umbrella","handbag",
"tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard",
"tennis racket","bench","traffic light","fire hydrant","stop sign",
"parking meter","mouse","remote","keyboard","microwave","oven",
"toaster","sink","refrigerator","book","clock","vase","scissors",
"teddy bear","hair drier","toothbrush",
}


@dataclass 
class Detection :
    bbox :tuple 
    centroid :tuple 
    area :float 
    label :str 
    raw_label :str 
    confidence :float 
    color :tuple =(0 ,220 ,80 )
    track_id :int =-1 
    source :str ="yolo"





class UnifiedDetector :
    """
    YOLOv8 on COCO, collapsed to clean display groups.
    Thermal detector is disabled by default — it produced 70+ noise boxes.
    """

    def __init__ (self ,
    yolo_model_size :str ="n",
    yolo_conf :float =0.40 ,
    iou_threshold :float =0.45 ,
    device :str ="cpu",
    use_thermal :bool =False ):

        self .conf =yolo_conf 
        self .iou =iou_threshold 
        self .device =device 
        self .use_thermal =use_thermal 
        self .model =None 
        self .tracker =CentroidTracker ()
        self ._load (yolo_model_size )

    def _load (self ,size :str ):
        try :
            from ultralytics import YOLO 
            self .model =YOLO (f"yolov8{size }.pt")
            self .model .fuse ()
            print (f"[YOLO] yolov8{size }.pt loaded — {len (COCO_CLASSES )} COCO classes")
        except ImportError :
            print ("[YOLO] Run: pip install ultralytics")
        except Exception as e :
            print (f"[YOLO] Load failed: {e }")


    def _prep (self ,frame :np .ndarray )->np .ndarray :
        img =frame .copy ()
        if img .dtype ==np .float32 or img .dtype ==np .float64 :
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

    def detect (self ,frame :np .ndarray ,
    is_thermal :bool =False )->List [Detection ]:
        if self .model is None :
            return []

        img =self ._prep (frame )
        h ,w =img .shape [:2 ]
        min_area =w *h *0.003 

        try :
            results =self .model (
            img ,conf =self .conf ,iou =self .iou ,
            verbose =False ,device =self .device ,
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
                cls_idx =int (box .cls [0 ])
                raw_label =(COCO_CLASSES [cls_idx ]
                if cls_idx <len (COCO_CLASSES )else "object")


                if raw_label in IGNORED_CLASSES :
                    continue 
                if raw_label not in CLASS_GROUP :
                    continue 

                display_label ,color =CLASS_GROUP [raw_label ]
                bw ,bh =x2 -x1 ,y2 -y1 
                area =float (bw *bh )


                if area <min_area :
                    continue 

                detections .append (Detection (
                bbox =(x1 ,y1 ,bw ,bh ),
                centroid =(x1 +bw //2 ,y1 +bh //2 ),
                area =area ,
                label =display_label ,
                raw_label =raw_label ,
                confidence =conf ,
                color =color ,
                source ="yolo",
                ))


        if self .use_thermal and is_thermal :
            thermal =self ._thermal_hotspots (frame ,min_area *4 )
            detections .extend (thermal )


        detections =self .tracker .update (detections )
        return detections 

    def _thermal_hotspots (self ,frame :np .ndarray ,
    min_area :float )->List [Detection ]:
        """
        Very conservative hotspot detector — only fires on very large,
        extremely bright regions (top 0.5% of pixels).
        Prevents the 70-box noise problem.
        """
        gray =frame .astype (np .float32 )
        if gray .ndim ==3 :
            gray =cv2 .cvtColor (gray ,cv2 .COLOR_BGR2GRAY )

        threshold =np .percentile (gray ,99.5 )
        binary =(gray >=threshold ).astype (np .uint8 )*255 

        k =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(9 ,9 ))
        binary =cv2 .morphologyEx (binary ,cv2 .MORPH_CLOSE ,k )
        binary =cv2 .morphologyEx (binary ,cv2 .MORPH_OPEN ,k )

        contours ,_ =cv2 .findContours (
        binary ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )

        dets =[]
        for cnt in contours :
            area =cv2 .contourArea (cnt )
            if area <min_area :
                continue 
            x ,y ,bw ,bh =cv2 .boundingRect (cnt )
            dets .append (Detection (
            bbox =(x ,y ,bw ,bh ),
            centroid =(x +bw //2 ,y +bh //2 ),
            area =area ,
            label ="Hotspot",
            raw_label ="hotspot",
            confidence =0.90 ,
            color =(0 ,0 ,255 ),
            source ="thermal",
            ))
        return dets 





class CentroidTracker :
    def __init__ (self ,max_disappeared :int =15 ,max_distance :float =80.0 ):
        self .next_id =0 
        self .objects ={}
        self .disappeared ={}
        self .max_gone =max_disappeared 
        self .max_dist =max_distance 

    def update (self ,detections :List [Detection ])->List [Detection ]:
        if not detections :
            for oid in list (self .disappeared ):
                self .disappeared [oid ]+=1 
                if self .disappeared [oid ]>self .max_gone :
                    self .objects .pop (oid ,None )
                    self .disappeared .pop (oid ,None )
            return detections 

        new_centroids =np .array ([d .centroid for d in detections ],dtype =np .float32 )

        if not self .objects :
            for i ,d in enumerate (detections ):
                self .objects [self .next_id ]=new_centroids [i ]
                self .disappeared [self .next_id ]=0 
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
                detections [c ].track_id =self .next_id 
                self .next_id +=1 

        for r in range (len (obj_ids )):
            if r not in used_rows :
                oid =obj_ids [r ]
                self .disappeared [oid ]+=1 
                if self .disappeared [oid ]>self .max_gone :
                    self .objects .pop (oid ,None )
                    self .disappeared .pop (oid ,None )

        return detections 





def draw_detections (frame :np .ndarray ,
detections :List [Detection ],
box_thickness :int =2 ,
font_scale :float =0.55 )->np .ndarray :
    """
    Draw clean bounding boxes.
    Label = display group name only (e.g. "Vehicle", "Person").
    No confidence score. No #track_id. Matches Image 3 style.
    """

    if frame .dtype !=np .uint8 :
        vis =cv2 .normalize (frame ,None ,0 ,255 ,cv2 .NORM_MINMAX ).astype (np .uint8 )
    else :
        vis =frame .copy ()

    if vis .ndim ==2 :
        vis =cv2 .cvtColor (vis ,cv2 .COLOR_GRAY2BGR )

    for d in detections :
        x ,y ,w ,h =[int (v )for v in d .bbox ]
        color =d .color 


        cv2 .rectangle (vis ,(x ,y ),(x +w ,y +h ),color ,box_thickness )


        label =d .label 
        (tw ,th ),baseline =cv2 .getTextSize (
        label ,cv2 .FONT_HERSHEY_SIMPLEX ,font_scale ,1 )


        ly =y -6 if y >th +10 else y +th +6 
        cv2 .rectangle (vis ,
        (x ,ly -th -4 ),
        (x +tw +6 ,ly +2 ),
        color ,-1 )

        text_color =(0 ,0 ,0 )
        cv2 .putText (vis ,label ,
        (x +3 ,ly -2 ),
        cv2 .FONT_HERSHEY_SIMPLEX ,font_scale ,
        text_color ,1 ,cv2 .LINE_AA )

    return vis 


def detections_to_json (detections :List [Detection ],
frame_id :str ="frame")->dict :
    return {
    "frame_id":frame_id ,
    "count":len (detections ),
    "detections":[
    {
    "label":d .label ,
    "raw_label":d .raw_label ,
    "confidence":round (d .confidence ,3 ),
    "bbox":list (d .bbox ),
    "centroid":list (d .centroid ),
    "area":round (d .area ,1 ),
    "source":d .source ,
    }
    for d in detections 
    ],
    }

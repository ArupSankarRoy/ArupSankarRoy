
from ultralytics import YOLO
import cv2
import numpy as np
import os
from classnames import YoloClassNames
from edgedetector import *

# def Points(events , x ,y , flags,param):
#     if events == cv2.EVENT_MOUSEMOVE:
#         print(f"x:{x} y:{y}")

# cv2.namedWindow("frame")
# cv2.setMouseCallback("frame",Points)

def video_show(path:str):

    cap = cv2.VideoCapture(path)
    model = YOLO("yolov8x.pt")
    width,height = (1000,576)
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)
        
        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf_ = float(box.conf[0])
                cls_ = YoloClassNames()[int(box.cls[0])]
                
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{cls_.capitalize()} | {round(conf_,2)}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
        
        frame = cv2.resize(frame,(width,height))
        processed_frame = process_image(frame,width,height)
    
        cv2.imshow("frame",processed_frame)
        if cv2.waitKey(1) &  0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = os.path.join(os.getcwd(),"upload","sample_001.mp4")
    video_show(video_path)
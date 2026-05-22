# -*- coding: utf-8 -*-

#!pip install ultralytics

import os
import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("best_R.pt") # best from YOLOv11m train

DirPath = './4' # direktori class KL
Files = os.listdir(DirPath)

for File in Files:
    imgPath = os.path.join(DirPath, File)
    print(imgPath)
    image=cv2.imread(imgPath)
    
    results = model.predict(image,conf=0.1)
    
    result = results[0]
    
    len(result.boxes)
    
    box = result.boxes[0]
    
    
    cards = box.xyxy[0].tolist()
    cards = [round(x) for x in cards]
    class_id = result.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)
    
    for box in reversed(result.boxes):
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        left, top, right, bottom = cords[0], cords[1], cords[2], cords[3]
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(),2)
        cx=int((left+right)/2)
        cy=int((top+bottom)/2)
        pt=[int(class_id),cx,cy]
    
        if (class_id=='1'):
            pt1=[cx,cy]
        elif (class_id=='2'):
            pt2=[cx,cy]
        elif (class_id=='3'):      
            pt3=[cx,cy]
        elif (class_id=='4'):
            pt4=[cx,cy]
        else:
            print("Error")
    
    x1=pt1
    x2=pt2
    x3=pt3
    x4=pt4
        
    xray1 = image
    pts1 = np.float32([[x1],[x2],[x3],[x4]]) # Coordinates that you want to Perspective Transform
    pts2 = np.float32([[26,60],[273,40],[46,245],[253,240]])  # Size of the Transformed Image #R knee  
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(xray1,M,(299,299))
    
    cv2.imwrite(f'./4_4T/4_4T_{File}',dst)# Direktori and file result

print('Finish...')  
# -*- coding: utf-8 -*-

#pip install "ultralytics<=8.3.40" supervision roboflow

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from IPython.display import display, Image
model = YOLO("yolo11m.pt")
model.train(data = '/home/informatika/workspace/source/SPM_UMBY/YL11_OAI_L/data.yaml', epochs = 100, optimizer='adam', batch=16, imgsz=640, lr0=0.001, lrf=0.001)

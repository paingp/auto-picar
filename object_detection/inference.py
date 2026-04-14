"""
inference.py

Perform inference on given image to detect toy road signs with YOLO model.
The model is capable of detecting the following:
   - stop sign
   - caution sign
   - crosswalk sign
   - lane lines (black tape)
"""


import psutil
import cv2
import torch
import math

from ultralytics import YOLO
from picamera2 import Picamera2

model = YOLO("yolo_float16.tflite", task="detect")

#cpu_stat = []
#mem_stat = []

cam = Picamera2()
cam.preview_configuration.main.size = (640, 480)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()
image = cam.capture_array()

#image = "output_images/6.jpg" #"dataset/lane/lane_169.jpg"   # 141 
result = model(image)[0]
annotated_frame = result.plot()

boxes = result.boxes
classes = result.boxes.cls.int()
print(type(classes))
if classes.numel() > 0:
    print(classes)
    signs_mask = classes != 2 
    if torch.any(signs_mask):
        xywhn_sign = result.boxes.xywhn[signs_mask]
        print(xywhn_sign)
        x = xywhn_sign[0][0]
        diff = x - 0.5
        #print(diff)
        #print(math.asin(diff))
    #print(boxes.xywhn[classes == 2])
    #print(boxes.xywhn[classes == 2][0][-1])

cv2.imshow("camera", annotated_frame)
cv2.waitKey(0)


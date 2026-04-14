"""
collect_images.py

Script used to collect images of road signs at various angles to train object detection model.
"""


import os
import re
import time

import cv2

from picamera2 import Picamera2, Preview
from picarx import Picarx

output_dir = "dataset/lane"
os.makedirs(output_dir, exist_ok=True)

existing_images = [f for f in os.listdir(output_dir) if re.match(r"l_\d+\.jpg", f)]
existing_numbers = [int(re.findall(r"\d+", f)[0]) for f in existing_images]
start_num = max(existing_numbers) + 1 if existing_numbers else 1

car = Picarx()

cam = Picamera2()
cam.preview_configuration.main.size = (640, 480)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

for num, angle in enumerate(range(-10, 10, 2)):
    car.set_cam_pan_angle(angle)
    time.sleep(0.2)    
    filename = f"{start_num + num:03d}.jpg"
    filepath = os.path.join(output_dir, filename)
    cam.capture_file(filepath)



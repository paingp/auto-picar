"""
detect_objects.py

Perform real time inference to detect road signs on stream of input images from camera.
Road signs model can detect:
   - Stop 
   - Caution
   - Crosswalk
"""


import psutil
import time

import cv2

from pathlib import Path
from picamera2 import Picamera2
from ultralytics import YOLO 

def export_model(torch_model):
    model = YOLO(torch_model)
    model.export(format='tflite')

def monitor_system(debug=False):
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    #disk_usage = psutil.disk_usage('/')

    if debug:
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory.percent}%")

    return (cpu_usage, memory.percent)

def get_average(darray):
    return (sum(darray) / len(darray))

cam = Picamera2()
'''
cam.preview_configuration.main.size = (640, 640)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
'''

config = cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
cam.configure(config)
#cam.set_controls({"ExposureTime": 25000})		# 10 ms
cam.start()

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "object_detection" / "yolo_float16.tflite"
model = YOLO(model_path, task='detect')

start = time.time()
inference_times = []
cpu_stat = []
memory_stat = []
images = 1

while True:
    frame = cam.capture_array()
    
    start_inf = time.time()
    results = model(frame)
    end_inf = time.time()
    inference_times.append((end_inf - start_inf) * 1000)
    cpu, mem = monitor_system(True)
    cpu_stat.append(cpu)
    memory_stat.append(mem)

    annotated_frame = results[0].plot()

    cv2.imshow("Camera", annotated_frame)

    if cv2.waitKey(10) == ord('q'):
        break
    images += 1
    print("Memory used:", psutil.virtual_memory().used / 1024**2, "MB")

cv2.destroyAllWindows()

end = time.time()

print(f"Time elapsed: {end - start} sec")
print(f"Images processed: {images}")
print(f"Avg. Inference time: {get_average(inference_times)}")
print(f"CPU Usage: {get_average(cpu_stat)}")
print(f"Memory Usage: {get_average(memory_stat)}")


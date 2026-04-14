"""
follow_signs.py

Script to test how car reacts to stop signs.
"""

import threading
import time

import cv2

from picarx import Picarx
from picamera2 import Picamera2
from ultralytics import YOLO
from robot_hat import TTS


class DriveSystem:
    def __init__(self, speed=25, dthresh=20):
        self.car = Picarx()
        self.speed = speed		# range: [-100, 100] 
        self.drive = True 
        self.distance = 0
        self.d_sign = 0
        self.dthresh = dthresh		# distance threshold to obstacle 

def drive(drsys):
    drsys.car.forward(drsys.speed)
    try:
        while drsys.drive:
            time.sleep(0.2)
            if drsys.d_sign:
                if drsys.d_sign <= 3:
                    drsys.car.stop()
                    drsys.drive = False
                drsys.d_sign -= 1.94
            print("drive?", drsys.drive)
    except KeyboardInterrupt:
        print("Interrupt caught in thread")
        drsys.car.stop()
    drsys.car.stop()

def avoid_obstacles(drsys):
    while drsys.drive:
        dist = drsys.car.ultrasonic.read()
        print("distance:", dist)
        if dist <= drsys.dthresh:
            print("Obstacle detected!")
            drsys.car.stop()
            drsys.drive = False
            drsys.distance = dist
        time.sleep(0.2)

def detect_sign(model, img, txt2speech, debug=False):
    result = model.predict(img)[0]
    classes = result.boxes.cls.int()
    if (classes.numel()):
        print(classes.item())
        obj_h = result.boxes.xywh[0, -1].item()      # object height 
        print("Object height:", obj_h)
        if classes.item() == 2:
            print("Stop sign detected!")
            if debug:
                txt2speech.say("Stop sign") 
            if obj_h <= 106.5:
                drsys.d_sign = 660 / obj_h
                print("Distance to sign:", drsys.d_sign)
            else:
                drsys.car.stop()
                drsys.drive = False
        elif classes.item() == 0:
            print("Caution detected")
            if debug:
                txt2speech.say("Caution")
            drsys.car.forward(5)
            drsys.speed = 10
        print([result.names[cls.item()] for cls in result.boxes.cls.int()])
        print(result.boxes.conf) 
    cv2.imshow("camera", img)
    cv2.waitKey(20)

def monitor_camera(drsys, cam, model, txt2speech, debug=False):
    while drsys.drive:
        frame = cam.capture_array()
        detect_sign(model, frame, txt2speech, debug)


if __name__ == "__main__":
    speed = 25
    drsys = DriveSystem(speed) 
    drsys.car.reset()

    cam = Picamera2()
    cam.preview_configuration.main.size = (640, 640)
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")
    cam.start()

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "object_detection" / "yolo_float16.tflite"
    model = YOLO(model_path, task='detect')
    txt2speech = TTS()

    debug = True
    thread_avoid_ob = threading.Thread(target=avoid_obstacles, args=[drsys])
    thread_cam = threading.Thread(target=monitor_camera, args=[drsys, cam, model, txt2speech, debug])
    thread_drive= threading.Thread(target=drive, args=[drsys])
    #process_drive = multiprocessing.Process(target=drive, args=[drsys])
    #process_avoid_ob = multiprocessing.Process(target=avoid_obstacles, args=[drsys])

    txt2speech.say("System start")

    '''
    frame = cam.capture_array()
    result = model.predict(frame)[0]
    annotated_frame = result.plot()
    cv2.imshow("camera", annotated_frame)
    cv2.waitKey(1000)
    classes = result.boxes.cls.int()
    if (classes.numel()):
        print(classes.item())
        if classes.item() == 2:
            print(result.boxes)
            print(result.boxes.xywhn[0, -1].item())
            #txt2speech.say("Stop sign")
    '''
    try:
        thread_cam.start()
        time.sleep(3)
        thread_drive.start()
        #thread_avoid_ob.start()

    except KeyboardInterrupt:            
        print("Interrupt from main thread")
        drsys.car.stop()
        thread_drive.join()
        #thread_avoid_ob.join()
        thread_cam.join()

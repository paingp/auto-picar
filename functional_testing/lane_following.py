"""
lane_detection.py

Performs lane detection and lane following using opencv.
"""


import logging
import math
import os
import threading
import time

import cv2
import numpy as np

from picarx import Picarx
from picamera2 import Picamera2
from robot_hat import TTS
from auto-picar import opencv_lane_detection as old 


lock = threading.Lock()
cam_ready = threading.Condition(lock) 

class DriveSystem:
    def __init__(self, speed=25, dthresh=20):
        self.car = Picarx()
        self.speed = speed		# range: [-100, 100] 
        self.drive = True
        self.distance = 0
        self.d_sign = 900 
        self.dthresh = dthresh		# distance threshold to obstacle 
        self.image = None 
        self.img_num = 0
        self.left_line = None 
        self.right_line = None 
        self.lane_offset = False

def steer_motor(drsys):
    if drsys.left_line is None and drsys.right_line is not None:
        print("Left lane line not detected")
        cv2.putText(drsys.image, "Left line?", (400, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        drsys.car.set_dir_servo_angle(-10)
    elif drsys.left_line is not None and drsys.right_line is None:
        print("Right lane line not detected")
        cv2.putText(drsys.image, "Right line?", (400, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        drsys.car.set_dir_servo_angle(10)
    elif drsys.left_line is not None and drsys.right_line is not None:
        midpoint = old.get_midpoint(drsys.left_line, drsys.right_line, drsys.image, drsys.img_num)

def drive(drsys, img_size, logger):
    center_x = img_size / 2		# current x direction car is headed
    thresh = 70
    try:
        while drsys.drive:
            #start = time.time()
            with cam_ready:
                while drsys.image is None:
                    cam_ready.wait()
                print("Drive start")
            drsys.car.forward(drsys.speed)
            if drsys.d_sign <= 3:
                drsys.car.stop()
                drsys.drive = False
            drsys.d_sign -= 1.94
            if drsys.left_line is None and drsys.right_line is not None:
                logger.info("Left lane line not detected")
                cv2.putText(drsys.image, "Left line?", (400, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
                if drsys.lane_offset is None:
                    drsys.car.set_dir_servo_angle(-10)
                else:
                    drsys.lane_offset = None
            elif drsys.left_line is not None and drsys.right_line is None:
                logger.info("Right lane line not detected")
                cv2.putText(drsys.image, "Right line?", (400, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
                if drsys.lane_offset is None:
                    drsys.car.set_dir_servo_angle(10)
                else:
                    drsys.lane_offset = None
            elif drsys.left_line is not None and drsys.right_line is not None:
                midpoint = old.get_midpoint(drsys.left_line, drsys.right_line, drsys.image, drsys.img_num)
                if drsys.car.dir_current_angle == 0:
                    dir_x = center_x
                else:
                    dir_x = center_x + drsys.speed * math.sin(math.radians(drsys.car.dir_current_angle))
                diff = midpoint - dir_x
                logger.info(f"midpoint: {midpoint:.3f}")
                logger.info(f"diff: {diff:.3f}")
                if diff > thresh:
                    steering_angle = 15 
                    print("Need to turn right")
                elif diff < -thresh:
                    steering_angle = -15
                    print("Need to turn left")
                else:
                    steering_angle = 0
                drsys.car.set_dir_servo_angle(steering_angle)
                cv2.putText(drsys.image, f"Steer: {steering_angle}", (400, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                drsys.lane_offset = diff 
            else:
                logger.info(f"Image {drsys.img_num}: No lane lines detected!")
                cv2.putText(drsys.image, "No lane", (400, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(f"output_images/{drsys.img_num}.jpg", drsys.image)
            drsys.car.forward(drsys.speed)
            logger.info(f"steering angle: {drsys.car.dir_current_angle}") 
            drsys.img_num += 1
            #end = time.time()
            #print(f"drive: {(end - start):.3f}")  
            time.sleep(0.1)
    except KeyboardInterrupt:
        drsys.car.stop()

def avoid_obstacles(drsys, logger):
    while drsys.drive:
        #start = time.time()
        dist = drsys.car.ultrasonic.read()
        logger.info(f"distance: {dist}")
        if dist <= drsys.dthresh:
            logger.info("Obstacle detected!")
            drsys.car.stop()
            drsys.drive = False
            drsys.distance = dist
        time.sleep(0.2)
        #end = time.time()
        #print(f"ultrasonic: {(end - start):.3f}") 

def detect_lane(frame):
    image_gray = old.preprocess_image(frame)
    min_val = 100	# threshold values to detect edge
    max_val = 200
    edges = cv2.Canny(image_gray, min_val, max_val)
    roi = old.extract_roi(edges)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength = 20, maxLineGap = 20)

    return old.detect_lines(roi, lines)

def lane_correction(car):
    while drsys.drive:
        start = time.time()
        gray_vals = car.get_grayscale_data()
        status = car.get_line_status(gray_vals)
        print("grayscale:", status) 
        if status[0] == 1:
            car.set_dir_servo_angle(20)
            print("Lane correction: turn right")
        elif status[2] == 1:
            car.set_dir_servo_angle(-20)
            print("Lane correction: turn left")
        #else:
        #    car.set_dir_servo_angle(0)
        end = time.time()
        print(f"lanec: {(end - start):.3f}") 
        time.sleep(0.2)

def monitor_camera(drsys, cam, txt2speech, logger):
    while drsys.drive:
        #start = time.time()
        with cam_ready:
            drsys.image = cam.capture_array()
            #cv2.imshow("Image", frame)
            #cv2.waitKey(1000)
            drsys.left_line, drsys.right_line = detect_lane(drsys.image)
            cam_ready.notify()
            logger.info("Image ready")
        #end = time.time()
        #print(f"cam: {(end - start):.3f}") 
        cam.stop()
        if drsys.img_num % 2 == 0:
            cam.set_controls({"ExposureTime": 10000})
        else:
            cam.set_controls({"ExposureTime": 30000})
        cam.start()
        time.sleep(0.1)


if __name__ == "__main__":
    speed = 10
    img_size = 640
    drsys = DriveSystem(speed) 
    drsys.car.set_dir_servo_angle(0)
    #drsys.car.reset()

    cam = Picamera2()
    '''
    #cam.preview_configuration.main.size = (img_size, img_size)
    #cam.preview_configuration.main.format = "RGB888"
    #cam.preview_configuration.align()
    #cam.configure("preview")
    '''
    config = cam.create_video_configuration(main={"size": (img_size, img_size)})
    cam.set_controls({"ExposureTime": 10000})
    cam.start()
    print(cam.camera_config)

    txt2speech = TTS()

    output_path = "output_images"
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print("Error deleting output images:", e)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("app.log", mode='w')

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    console_format = logging.Formatter("%(levelname)s: %(message)s")
    file_format = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    thread_avoid_ob = threading.Thread(target=avoid_obstacles, args=[drsys, logger])
    thread_cam = threading.Thread(target=monitor_camera, args=[drsys, cam, txt2speech, logger])
    thread_drive = threading.Thread(target=drive, args=[drsys, img_size, logger])
    #thread_lc = threading.Thread(target=lane_correction, args=[drsys.car])

    txt2speech.say("System start")

    try:
        thread_cam.start()
        #thread_lc.start()
        #time.sleep(2.4)
        thread_drive.start()
        thread_avoid_ob.start()

    except KeyboardInterrupt:            
        print("Interrupt from main thread")
        drsys.car.stop()
        thread_drive.join()
        thread_avoid_ob.join()
        #thread_lc.join()
        thread_cam.join()

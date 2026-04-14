"""
auto_drive.py

Multi-threaded program that runs the autonomous algorithm to drive the car. The smart algorithm is capable of performing the following:
   - Control steering to drive within lane lines (marked with black tape on carpet)
   - Detect and obey road signs

Behavior when obeying different road signs:
   - Stop: come to a permanent stop
   - Cuation: slow down
   - Crosswalk: stop temporarily and proceed slowly
"""


import logging
import math
import os
import threading
import time

import cv2
import numpy as np

import opencv_lane_detection as old
import ai_lane_detection as aild

from picarx import Picarx
from picamera2 import Picamera2
from robot_hat import TTS
from ultralytics import YOLO


lock = threading.Lock()
cam_ready = threading.Condition(lock)

def stop_car(car):
    car.stop()
    car.set_dir_servo_angle(0)

class DriveSystem:
    def __init__(self, speed=25, dthresh=20, model=None):
        self.car = Picarx()
        self.speed = speed		# range: [-100, 100] 
        self.steering_angle = 0
        self.drive = True
        self.distance = 0
        self.d_sign = 900 
        self.dthresh = dthresh		# distance threshold to obstacle 
        self.image = None 
        self.img_num = 0
        self.model = model
        self.left_line = None 
        self.right_line = None 
        self.lane_offset = 0
        self.road_sign = 0
        self.sign_det_time = None

def steer_motor(drsys, center_x, logger):
    #steering_angle = 0
    k = 0.02	# exponential decay rate for turn

    if drsys.left_line is None and drsys.right_line is None:
        logger.info("Didn't detect any lane lines!")
        cv2.putText(drsys.image, "No lines", (400, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 

    elif drsys.left_line is None:
        logger.info("Left lane line not detected")
        cv2.putText(drsys.image, "Left line?", (400, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.circle(drsys.image, (int(drsys.right_line[0]), int(drsys.right_line[1])), 5, (255, 0, 0), -1)
        if drsys.right_line[0] <= center_x:
            drsys.steering_angle = -16 
        else:
            diff = float(drsys.right_line[0]) - center_x
            print("center_x:", center_x)
            print("right line:", drsys.right_line)
            print("diff:", diff)
            drsys.steering_angle = -round(24 * (1 - (diff / 280))) 

    elif drsys.right_line is None:
        logger.info("Right lane line not detected")
        cv2.putText(drsys.image, "Right line?", (400, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.circle(drsys.image, (int(drsys.left_line[0]), int(drsys.left_line[1])), 5, (255, 0, 0), -1)
        if drsys.left_line[0] >= center_x:
            drsys.steering_angle = 16 
        else:
            diff = center_x - float(drsys.left_line[0])
            print("center_x:", center_x)
            print("left line:", drsys.left_line)
            print("diff:", diff)
            #drsys.steering_angle = round(24 * math.exp(-k * diff)
            drsys.steering_angle = round(24 * (1 - (diff / 280))) 

    else:
        if drsys.img_num:
            midpoint = float(drsys.left_line[0] + drsys.right_line[0]) / 2
        else:
            print("------Traditional CV Lane detection------")
            midpoint = old.get_midpoint(drsys.left_line, drsys.right_line, drsys.image, annotate=True)
        cv2.circle(drsys.image, (int(midpoint), int(drsys.left_line[1])), 5, (0, 255, 0), -1)
        if drsys.car.dir_current_angle == 0:
            dir_x = center_x
        else:
            dir_x = center_x + drsys.speed * math.sin(math.radians(drsys.car.dir_current_angle))
        drsys.lane_offset = midpoint - dir_x
        logger.info(f"midpoint: {midpoint:.3f}")
        logger.info(f"offset: {drsys.lane_offset:.3f}")

        gain = 0.5
        norm_offset = drsys.lane_offset / 240 
        if abs(drsys.lane_offset) >= 48:
            #drsys.steering_angle = math.floor(gain * drsys.lane_offset)
            drsys.steering_angle = round(float(norm_offset) * 24)
            if abs(drsys.steering_angle - drsys.car.dir_current_angle) < 1:
                drsys.steering_angle = drsys.car.dir_current_angle
        else:
            drsys.steering_angle = 0

    drsys.car.set_dir_servo_angle(drsys.steering_angle)
    cv2.putText(drsys.image, f"Steer: {drsys.steering_angle}", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    '''
    if drsys.lane_offset <= -48:
        drsys.car.set_dir_servo_angle(-5)
    elif drsys.lane_offset >= 48:
        drsys.car.set_dir_servo_angle(5)
    else:
        drsys.car.set_dir_servo_angle(0)
    '''

def detect_sign(classes, boxes):
    active_sign = 0
    sign_height = 0	# center x-coordinate of sign

    if 0 in classes:
        caution_mask = classes == 0
        sign_height = boxes[caution_mask][0][-1]
        if sign_height > 0.8:
            active_sign = 3 

    elif 1 in classes:
        cross_mask = classes == 1
        sign_height = boxes[cross_mask][0][-1]
        if sign_height > 0.1:
            active_sign= 2 

    elif 3 in classes:	# Stop sign
        stop_mask = classes == 3
        sign_height = boxes[stop_mask][0][-1]
        if sign_height > 0.1:
            active_sign = 1 

    if sign_height:
        print("sign height:", sign_height)

    return active_sign 

def obey_sign(sign_id, drsys, logger, txt2speech, debug):
    print("sign ID:", sign_id)
    match sign_id:
        case 1:
            logger.info(f"Stop sign detected!")
            if debug:
                txt2speech.say("stop") 
            drsys.car.stop()
            drsys.drive = False
        case 2:
            logger.info(f"Crosswalk detected")
            if debug:
                txt2speech.say("crosswalk") 
            drsys.car.stop()
            drsys.speed = 0
        case 3:
            logger.info(f"Caution sign detected")
            if debug:
                txt2speech.say("caution") 
            drsys.speed = 5 
            drsys.car.forward(drsys.speed)
            print("Set speed to", drsys.speed)
        case _:
            print(f"Sign {sign_id} detected")

def drive_thread(drsys, img_size, logger, output_path, txt2speech, cam):
    center_x = img_size / 2		# current x direction car is headed
    thresh = 70
    sign_delay = 1e-6
    id_to_sign = {1: "stop", 2: "crosswalk", 3: "caution"}
    try:
        while drsys.drive:
            #start = time.time()
            with cam_ready:
                #if drsys.image is None: 
                cam_ready.wait()
            logger.info(f"Image {drsys.img_num}")
            #drsys.car.forward(drsys.speed)
            #print("Speed:", drsys.speed)
            result = drsys.model(drsys.image, conf=0.5)[0]
            
            #cv2.imwrite(f"{output_path}/o{drsys.img_num}.jpg", drsys.image)
            if drsys.img_num:
                #drsys.left_line, drsys.right_line = aild.detect_lines(result)
                left_line, right_line = aild.detect_lines(result)
                if drsys.left_line is None and drsys.right_line is not None:
                    if left_line is not None and right_line is not None:
                        drsys.left_line, drsys.right_line = left_line, right_line
                elif drsys.left_line is not None and drsys.right_line is None:
                    if left_line is not None and right_line is not None:
                        drsys.left_line, drsys.right_line = left_line, right_line
                else:
                    drsys.left_line, drsys.right_line = left_line, right_line
                #print("Left line:", drsys.left_line)
                #print("Right line:", drsys.right_line)
                drsys.image = result.plot()
            else:
                print("------Traditional CV Lane detection------")
                drsys.left_line, drsys.right_line = old.detect_lane(drsys.image) 

            classes = result.boxes.cls.int()
            if classes.numel() > 0:
                active_sign = detect_sign(classes, result.boxes.xywhn)
                if active_sign:
                    drsys.road_sign = active_sign
                    logger.info(f"Active sign: {id_to_sign[active_sign]} ({active_sign})")
                    if active_sign == 3:
                        txt2speech.say("caution") 
                    drsys.sign_det_time = time.time()
            if drsys.sign_det_time is not None:
                time_passed = time.time() - drsys.sign_det_time
                print("time since sign:", time_passed)
                if time_passed > sign_delay:
                    logger.info(f"Obey road sign")
                    obey_sign(drsys.road_sign, drsys, logger, txt2speech, debug=False)
                    time.sleep(1)
                    drsys.speed = 10
                    drsys.road_sign = 0
                    # Need to recompute lane lines due to delay
                    logger.info("Recompute lane lines")
                    #cv2.imwrite(f"{output_path}/c{drsys.img_num}.jpg", drsys.image)
                    result = drsys.model(drsys.image, conf=0.5)[0]
                    drsys.left_line, drsys.right_line = aild.detect_lines(result)
                #drsys.image = cam.capture_array()

            if drsys.drive:
                steer_motor(drsys, center_x, logger)
                drsys.car.forward(drsys.speed)
            
            cv2.imwrite(f"{output_path}/{drsys.img_num}.jpg", drsys.image)
            logger.info(f"steering angle: {drsys.car.dir_current_angle}") 
            drsys.img_num += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_car(drsys.car)

def avoid_obstacles(drsys, logger):
    try:
        while drsys.drive:
            dist = drsys.car.ultrasonic.read()
            logger.info(f"distance: {dist}")
            if dist <= drsys.dthresh:
                logger.info("Obstacle detected!")
                drsys.car.stop()
                drsys.drive = False
                drsys.distance = dist
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_car(drsys.car)

def detect_lane(frame):
    image_gray = lane_detection.preprocess_image(frame)
    edges = lane_detection.adaptive_canny(image_gray)
    roi = lane_detection.extract_roi(edges)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength = 20, maxLineGap = 20)

    return lane_detection.detect_lines(roi, lines)

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

def capture_thread(drsys, cam, logger):
    try:
        while drsys.drive:
            #start = time.time()
            with cam_ready:
                drsys.image = cam.capture_array()
                cam_ready.notify()
            #logger.info(f"Image {drsys.img_num} ready")
            #end = time.time()
            #print(f"cam: {(end - start):.3f}") 
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_car(drsys.car)


if __name__ == "__main__":
    speed = 10
    img_size = (640, 480)
    model = YOLO("object_detection/yolo_float16.tflite", task="detect")
    drsys = DriveSystem(speed, model=model) 
    drsys.car.set_dir_servo_angle(0)
    #drsys.car.set_cam_tilt_angle(-10)
    #drsys.car.reset()
     
    '''
    def signal_handler(sig, frame):
        drsys.car.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    '''

    cam = Picamera2()
    cam.preview_configuration.main.size = (img_size[0], img_size[1])
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")
    #config = cam.create_video_configuration(main={"size": (img_size, img_size), "format": "RGB888"})
    #cam.set_controls({"ExposureTime": 20000})
    cam.start()
    #print(cam.camera_config)

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

    avoid_ob_thread = threading.Thread(target=avoid_obstacles, args=[drsys, logger])
    cam_thread = threading.Thread(target=capture_thread, args=[drsys, cam, logger])
    drive_thread = threading.Thread(target=drive_thread, args=[drsys, img_size[0], logger, output_path, txt2speech, cam])
    #thread_lc = threading.Thread(target=lane_correction, args=[drsys.car])

    txt2speech.say("Ama yay")

    try:
        cam_thread.start()
        #thread_lc.start()
        #time.sleep(2.4)
        drive_thread.start()
        avoid_ob_thread.start()

    except KeyboardInterrupt:            
        print("Interrupt from main thread")
        stop_car(drsys.car)
        drive_thread.join()
        #avoid_ob_thread.join()
        #thread_lc.join()
        cam_thread.join()

import logging
import os
import sys
import threading
import time
import cv2

sys.path.insert(0, "/home/paing/picar-x/picarx")

from picarx import Picarx
from picamera2 import Picamera2

class DriveSystem:
    def __init__(self, speed=10, dthresh=10, model=None):
        self.lock = threading.Lock()
        self.car = Picarx()
        self.speed = speed		# range: [-100, 100] 
        self.steering_angle = 0
        self.drive = True
        self.dthresh = dthresh		# distance threshold to obstacle 
        self.image = None
        self.img_num = 0
        self.model = model
        self.time_taken = 0
        self.img_ls = []

    def set_speed(self, speed):
        with self.lock:
            self.speed = speed

    def set_steering_angle(self, steering_angle):
        with self.lock:
            self.steering_angle = steering_angle
    
    def set_drive_flag(self, drive):
        with self.lock:
            self.drive = drive

    def set_image(self, image):
        with self.lock:
            self.image = image

    def set_img_num(self, img_num):
        with self.lock:
            self.img_num = img_num

    def get_car(self):
        with self.lock:
            return self.car 

    def get_speed(self):
        with self.lock:
            return self.speed 

    def get_steering_angle(self):
        with self.lock:
            return self.steering_angle 

    def get_drive_flag(self):
        with self.lock:
            return self.drive 

    def get_image(self):
        with self.lock:
            return self.image 

    def get_img_num(self):
        with self.lock:
            return self.img_num

    def stop_car(self):
        self.set_drive_flag(False)
        self.set_speed(0)
        self.get_car().stop()
        self.get_car().set_dir_servo_angle(0)

def init_grayscale_module():
    init_state = [0, 0, 0]
    current_state = [0, 0, 0]

    while current_state == init_state:
        gm_val_list = drsys.get_car().get_grayscale_data()
        gm_state = drsys.get_car().get_line_status(gm_val_list)
        current_state = gm_state

def readjust_car(last_state:list, offset:int):
    if last_state[1] == 1:
        drsys.set_steering_angle(0)
    elif last_state[0] == 1:
        drsys.set_steering_angle(offset)
    elif last_state[2] == 1:
        drsys.set_steering_angle(-offset)

    drsys.get_car().set_dir_servo_angle(drsys.get_steering_angle())
    drsys.get_car().forward(drsys.get_speed())

def track_line(drsys:DriveSystem, logger):
    last_state = [0, 0, 0]

    while not exit_event.is_set() and drsys.get_drive_flag():
        gm_data = drsys.get_car().get_grayscale_data()
        gm_state = drsys.get_car().get_line_status(gm_data)
        offset = 20

        logger.debug(f"gm_data: {gm_data}")
        logger.debug(f"gm_state: {gm_state}")

        if gm_state[1] == 1:
            last_state = [0, 1, 0]
            drsys.set_steering_angle(0)
            logger.debug("Car going straight")
        elif gm_state[0] == 1:
            last_state = [1, 0, 0]
            drsys.set_steering_angle(-offset)
            logger.debug("Car turning left")
        elif gm_state[2] == 1:
            last_state = [0, 0, 1]
            drsys.set_steering_angle(offset)
            logger.debug("Car turning right")
        else:
            # check if track is stuck in between 2 sensors
            readjust_car(last_state, offset)
            time.sleep(0.1)

            # check if track has actually ended
            gm_data = drsys.get_car().get_grayscale_data()
            gm_state = drsys.get_car().get_line_status(gm_data)
            if gm_state == [0, 0, 0]:
                drsys.stop_car()
                logger.debug("Car stopping")

        drsys.get_car().set_dir_servo_angle(drsys.get_steering_angle())
        drsys.get_car().forward(drsys.get_speed())
        time.sleep(0)

def avoid_obstacles(drsys, logger):
    while not exit_event.is_set() and drsys.get_drive_flag():
        dist = drsys.get_car().ultrasonic.read()
        logger.info(f"distance: {dist}")
        if dist <= drsys.dthresh:
            logger.info("Obstacle detected!")
            drsys.stop_car()
        time.sleep(0.1)

def capture_image(drsys, cam, logger):
    start = time.time()

    while not exit_event.is_set() and drsys.get_drive_flag():
        if (drsys.get_img_num() % 15 == 0): drsys.img_ls.append(cam.capture_array())
        drsys.set_img_num(drsys.get_img_num() + 1)
        #logger.info(f"Image {drsys.img_num} ready")
        #time.sleep(0.1)
    
    end = time.time()
    drsys.time_taken = end - start

if __name__ == '__main__':
    drsys = DriveSystem()
    img_size = (640, 480)
    cam = Picamera2()
    cam.preview_configuration.main.size = (img_size[0], img_size[1])
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")
    cam.start()

    cam.set_controls({
    "AeEnable": False,
    "ExposureTime": 50000,
    })
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    
    init_grayscale_module()

    exit_event = threading.Event()

    drive_thread = threading.Thread(target=track_line, args=[drsys, logger])
    avoid_ob_thread = threading.Thread(target=avoid_obstacles, args=[drsys, logger])
    capture_thread = threading.Thread(target=capture_image, args=[drsys, cam, logger])

    drive_thread.start()
    avoid_ob_thread.start()
    capture_thread.start()

    while drsys.get_drive_flag():
        try:
            time.sleep(0.01)
        except KeyboardInterrupt:
            exit_event.set()

    drsys.stop_car()
    drive_thread.join()
    avoid_ob_thread.join()
    capture_thread.join()

    print(f"Num images: {drsys.get_img_num()}")
    print(f"FPS: {drsys.get_img_num() / drsys.time_taken}")

    for i in range(0, 25, 5):
        cv2.imwrite(f"img_{i}.jpg", drsys.img_ls[i])


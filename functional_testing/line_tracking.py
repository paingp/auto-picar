import logging
import threading
import time
import sys

sys.path.insert(0, "/home/paing/picar-x/picarx/")

from picarx import Picarx

class DriveSystem:
    def __init__(self, speed=10, dthresh=20, model=None):
        self.car = Picarx()
        self.speed = speed		# range: [-100, 100] 
        self.steering_angle = 0
        self.drive = True

def stop_car(car):
    car.stop()
    car.set_dir_servo_angle(0)

def track_line(drsys: DriveSystem, logger): 
    while (not exit_event.is_set() and drsys.drive):
        gm_data = drsys.car.get_grayscale_data()
        gm_state = drsys.car.get_line_status(gm_data)
        offset = 20

        if gm_state[1] == 1:
            drsys.steering_angle = 0
            logger.debug("Car going straight")
        elif gm_state[0] == 1:
            drsys.steering_angle = -offset
            logger.debug("Car turning left")
        elif gm_state[2] == 1:
            drsys.steering_angle = offset
            logger.debug("Car turning right")
        else:
            drsys.drive = False
            logger.debug("Car stopping")

        if drsys.drive:
            drsys.car.set_dir_servo_angle(drsys.steering_angle)
            drsys.car.forward(drsys.speed)
        else:
            stop_car(drsys.car)

if __name__=='__main__':
    drsys = DriveSystem()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    
    exit_event = threading.Event()

    last_state = [0, 0, 0]
    while True:
        gm_val_list = drsys.car.get_grayscale_data()
        gm_state = drsys.car.get_line_status(gm_val_list)
        print("outHandle gm_val_list: %s, %s"%(gm_val_list, gm_state))
        currentSta = gm_state
        if currentSta != last_state:
            break

    drive_thread = threading.Thread(target=track_line, args=[drsys, logger])
    drive_thread.start()

    while drsys.drive:
        try: time.sleep(0.01)
        except KeyboardInterrupt:
            exit_event.set()
            drive_thread.join()
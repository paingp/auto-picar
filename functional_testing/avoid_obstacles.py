"""
avoid_obstacles.py

Car keeps driving forward until it detects an obstacle and stops before the obstacle.
"""


import threading
import time
from picarx import Picarx


class DriveSystem:
    def __init__(self, speed=50, dthresh=20):
        self.car = Picarx()
        self.speed = speed		# range: [-100, 100] 
        self.drive = True 
        self.distance = 0
        self.dthresh = dthresh		# distance threshold to obstacle 

def avoid_obstacles_thread(drsys):
    while drsys.drive:
        dist = drsys.car.ultrasonic.read()
        print("distance:", dist)
        if dist <= drsys.dthresh:
            drsys.car.forward(0)
            drsys.drive = False 
            drsys.distance = dist
        time.sleep(0.2)
            
def drive_thread(drsys):
    drsys.car.forward(drsys.speed)
    while drsys.drive:
        time.sleep(0.1)
    print("Obstacle detected!")


if __name__ == "__main__":
    speed = 20
    distance_limit = 20
    drsys = DriveSystem(speed, distance_limit)

    drive_thread = threading.Thread(target=drive_thread, args=[drsys])
    avoid_ob_thread = threading.Thread(target=avoid_obstacles_thread, args=[drsys])

    drive_thread.start()
    avoid_ob_thread.start()

    drive_thread.join()
    avoid_ob_thread.join()

"""
manual_drive.py

Drive robot car in real time using keyboard input.

Contorls:
   w - forward
   a - left
   s - back
   d - right
   any other key - stop and exit control
"""

import time

from sshkeyboard import listen_keyboard, stop_listening
from picarx import Picarx

class CarController:
    def __init__(self):
        self.car = Picarx()

    def on_key_press(self, key):
        if key == 'w':
            self.car.set_dir_servo_angle(0)
            self.car.forward(15)
        elif key == 's':
            self.car.set_dir_servo_angle(0)
            self.car.backward(15)
        elif key == 'a':
            self.car.set_dir_servo_angle(-15)
            self.car.forward(10)
        elif key == 'd':
            self.car.set_dir_servo_angle(15)
            self.car.forward(10)
        else:
            self.stop()
            stop_listening()

    def stop(self):
        self.car.stop()
        self.car.set_dir_servo_angle(0)

if __name__ == "__main__":
    try:
        controller = CarController()
        listen_keyboard(on_press=controller.on_key_press)

    except KeyboardInterrupt:
        controller.stop() 

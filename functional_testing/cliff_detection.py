"""
cliff_detection.py

Detect edges with vertical drops and have robot avoid from driving off a cliff.
"""


from picarx import Picarx
from time import sleep


car = Picarx()
car.set_cliff_reference([200, 200, 200])

current_state = None
px_power = 10
offset = 20
last_state = "safe"

if __name__=='__main__':
    try:
        car.forward(50)
        while True:
            gm_val_list = car.get_grayscale_data()
            gm_state = car.get_cliff_status(gm_val_list)
            print("cliff status is:  %s"%gm_state)

            if gm_state is False:
                state = "safe"
            else:
                state = "danger"   
                car.stop()
                if last_state == "safe":
                    sleep(0.1)
                else:
                    break
            last_state = state

    finally:
        car.stop()
        sleep(0.5)
        car.backward(50)
        sleep(0.3)
        car.stop()


import time
import cv2

from picamera2 import Picamera2, Preview
from picarx import Picarx

car = Picarx()
#car.reset()

cam = Picamera2()
cam.preview_configuration.main.size = (640, 480)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")

#config = cam.create_video_configuration(main={"size": (640, 480)})
#cam.configure(config)
#cam.set_controls({"ExposureTime": 10000})
#print(cam.camera_controls)

cam.start()

car.set_cam_pan_angle(10)
time.sleep(0.2)
frame = cam.capture_array()
#metadata = cam.capture_metadata()
#print(metadata)
cv2.imshow("image", frame)
cv2.waitKey(0)

import sys
import builtins
import subprocess
import os

import cv2
import numpy as np

from libcamera import controls

from picamera2 import Picamera2
from picarx import Picarx
import time

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)	# pixels > 40 set to 255 (white)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    print("var:", variance)
    '''
    black_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 
                                          blockSize=15, C=2)

    # Sobel X gradient filter to detect vertical lines
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.uint8(np.clip(np.absolute(sobelx) * 255.0 / np.max(np.absolute(sobelx)), 0, 255))
    _, grad_mask = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_and(black_mask, grad_mask)
    '''
    
    return gray 

def adaptive_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def extract_roi(frame):
    height, width = frame.shape
    mask = np.zeros_like(frame)

    polygon = np.array([[(0, height), (100, height // 2), (width - 100, height // 2), (width, height)]], np.int32)
    #polygon = np.array([[(0, height), (0, height // 2), (width, height // 2), (width, height)]], np.int32)
    img_copy = np.copy(frame)
    #cv2.polylines(img_copy, [polygon], True, (255, 255, 255))
    cv2.fillPoly(mask, polygon, 255)
    #cv2.imshow("roi", img_copy)
    #cv2.waitKey(2000)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

def detect_lines(image, lines, debug=False):
    if lines is None:
        return (None, None)
    left_lines = []
    right_lines = []
    slopes = []
    #print("Lines detected: ", len(lines))
    for line in lines:
        x1, y1, x2, y2 = line[0] 
        if x1 != x2 and (abs(y2 - y1) >= 10):
            slope = (y2 - y1) / (x2 - x1)
            if (abs(slope) > 0.5):
                if slope < 0:
                    left_lines.append(line) 
                else:
                    right_lines.append(line)
                if debug:
                    #print(line)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 5)
                    cv2.imshow("image", image)
                    cv2.waitKey(1000)

    print("left lines:", len(left_lines))
    print("right lines:", len(right_lines))
    left_line = None
    right_line = None

    if left_lines:
        left_lines = np.array(left_lines)
        left_line = left_lines[np.argmax(left_lines[:, :, 1])][0]

    if right_lines:
        right_lines = np.array(right_lines)
        right_line = right_lines[np.argmax(right_lines[:, :, 3])][0]

    return (left_line, right_line)

def get_midpoint(left_line, right_line, image, annotate=False):
    lx1, ly1, lx2, ly2 = left_line	#[0]
    rx1, ry1, rx2, ry2 = right_line	#[0]
    m_l = (ly2 - ly1) / (lx2 - lx1)
    b_l = ly2 - m_l * lx2
    m_r = (ry2 - ry1) / (rx2 - rx1)
    b_r = ry1 - m_r * rx1
    #print(f"left: y = {m_l:.2f}x + {b_l:.2f}")
    #print(f"right: y = {m_r:.2f}x + {b_r:.2f}")
    
    y = image.shape[0] * 3 / 4 
    x_l = (y - b_l) / m_l
    x_r = (y - b_r) / m_r
    midpoint = (x_l + x_r) / 2
    #midpoint = (left_line[0][2] + right_line[0][0]) / 2
    if annotate:
        #cv2.line(image, (lx1, ly1), (lx2, ly2), (0, 0, 255), 5)
        #cv2.line(image, (rx1, ry1), (rx2, ry2), (0, 0, 255,), 5)
        cv2.circle(image, (int(midpoint), int(y)), 5, (0, 0, 255), -1)
    return midpoint

def detect_lane(frame):
    image_gray = preprocess_image(frame)
    min_val = 100	# threshold values to detect edge
    max_val = 200
    edges = cv2.Canny(image_gray, min_val, max_val)
    roi = extract_roi(edges)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength = 20, maxLineGap = 20)

    return detect_lines(roi, lines)

def _logging_popen(*args, **kwargs):
    print("[subprocess.Popen called]:", args[0])
    return _original_subprocess(*args, **kwargs)


#if __name__ == "__main__":
def main():
    #_original_subprocess = subprocess.Popen
    #subprocess.Popen = _logging_popen
    debug = False
    if len(sys.argv) > 1:
        if sys.argv[1]:
            debug = True 

    cam = Picamera2()
    cam.preview_configuration.main.size = (640, 640)
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")

    #config = cam.create_video_configuration(main={"size": (640, 480)})
    #cam.configure(config)
    #cam.set_controls({"ExposureTime": 20000})		# 10 ms
    cam.start()

    car = Picarx()
    #car.reset()
    time.sleep(0.2)

    image = cam.capture_array()
    #metadata = cam.capture_metadata()
    #print(metadata)
    #image = cv2.imread("output_images/0.jpg")

    img_gray = preprocess_image(image)
    cv2.imshow("prep", img_gray)
    cv2.waitKey(2000)
    '''
    min_val = 100	# threshold values to detect edge
    max_val = 200
    edges = cv2.Canny(img_gray, min_val, max_val)
    '''
    edges = adaptive_canny(img_gray)
    roi = extract_roi(edges)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength = 20, maxLineGap = 20) #10)
    left_line, right_line = detect_lines(roi, lines, debug=True)
    cv2.imshow("edges", roi)
    cv2.waitKey(2000)
    if debug:
        cv2.imwrite("ld0.jpg", image)
        cv2.imwrite("ld1.jpg", roi)

    print(left_line)
    print(right_line)

    if left_line is not None and right_line is not None:
        midpoint = get_midpoint(left_line, right_line, image, annotate=True)
        print("lane center:", midpoint)
        mid = roi.shape[1] / 2
        print("diff:", midpoint - mid)
    elif left_line is None and right_line is not None:
        print("Can't detect left lane line")
    elif left_line is not None and right_line is None:
        print("Can't detect right lane line")
    else:
        print("Can't detect lane lines")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    if debug:
        cv2.imwrite("ld2.jpg", image)

if __name__ == "__main__":
    cam = Picamera2()
    cam.preview_configuration.main.size = (640, 640)
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")

    cam.start()
    frame = cam.capture_array()
    left_line, right_line = detect_lane(frame)
    print(left_line)
    print(right_line)

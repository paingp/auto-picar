"""
ai_lane_detection.py

Use YOLO model to detect lane lines. 
"""


import psutil
import cv2
import torch

from picamera2 import Picamera2
from ultralytics import YOLO

def detect_lines(result):
    """
    Detect left and right lane lines from YOLO result object.

    Parameters:
        result: An object with attributes:
            - boxes.cls: Tensor of class indices
            - boxes.xywh: Tensor of bounding boxes in [cx, cy, w, h]
            - boxes.xyxy: Tensor of bounding boxes in [x1, y1, x2, y2]
            - orig_shape: Tuple (height, width) of original image

    Returns:
        (left_line, right_line): Tuple of tensors or None for left and right lines
    """
    left_line = None
    right_line = None

    # Filter class 2 (lane lines)
    classes = result.boxes.cls.int()
    lines_mask = classes == 2

    if torch.any(lines_mask):
        xywh_all = result.boxes.xywh[lines_mask]
        xyxy_all = result.boxes.xyxy[lines_mask]

        # Filter out boxes with center_y < 0.4
        center_y = xywh_all[:, 1]
        valid_mask = center_y >= 0.4
        xywh = xywh_all[valid_mask]
        xyxy = xyxy_all[valid_mask]

        if xywh.size(0) == 0:
            return None, None

        # Sort by bottom y (xyxy[:, 3]) descending
        sorted_indices = torch.argsort(xyxy[:, 3])
        xywh = xywh[sorted_indices]
        xyxy = xyxy[sorted_indices]

        center_x = result.orig_shape[1] / 2  # image width / 2

        if xywh.size(0) >= 2:
            # Assign left and right based on x-coordinate
            first_x = xyxy[0][0]
            second_x = xyxy[1][0]

            if xyxy[0][0] < xyxy[1][0]:
                left_line = xywh[0]
                if xywh[1][1] >= xyxy[0][1]:
                     right_line = xywh[1] 
            else:
                right_line = xywh[0]
                if xywh[1][1] >= xyxy[0][1]:
                    left_line = xywh[1]
        else:
            # Only one line detected -- decide left or right
            if xywh[0][0] < center_x:
                left_line = xywh[0]
            else:
                right_line = xywh[0]

    return left_line, right_line


def detect_lines1(result):
    left_line = None
    right_line = None

    classes = result.boxes.cls.int()
    if classes.numel() > 0:
        if 2 in classes:
            lines_mask = classes == 2
            boxes_xyxy = result.boxes.xyxy[lines_mask]		#.cpu()
            boxes_xywh = result.boxes.xywh[lines_mask]		#.cpu()
            lines_xyxy = boxes_xyxy
            lines_xywh = boxes_xywh
            #print(lines_xyxy)
            #print(lines_xywh)
            # Sort boxes for lane lines in order of descending y value (bottom right)
            sorted_indices = torch.argsort(lines_xyxy[:, 3]) #, descending=True)
            lines_xyxy = lines_xyxy[sorted_indices]
            lines_xywh = lines_xywh[sorted_indices]
            #print(lines_xyxy[sorted_indices]) 
            #print(lines_xywh[sorted_indices])
            #lines = sorted(lines, key=lamba b: b[1], reverse=True)
            #lines = sorted(lines, key=lambda b: b[0])
            center_x = result.orig_shape[1] / 2
            if torch.sum(lines_mask) >= 2:
                if lines_xyxy[0][0] > lines_xyxy[1][0]:
                    right_line = lines_xywh[0]
                    if lines_xywh[1][1] >= lines_xyxy[0][1]:
                        left_line = lines_xywh[1]
                else:
                    left_line = lines_xywh[0]
                    if lines_xywh[1][1] >= lines_xyxy[0][1]:
                        right_line = lines_xywh[1]
            else:
                if lines_xywh[0][0] < center_x:
                    left_line = lines_xywh[0]
                else:
                    right_line = lines_xywh[0]
            
    return (left_line, right_line)

if __name__ == "__main__":
    #'''
    cam = Picamera2()
    cam.preview_configuration.main.size = (640, 480)
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.start()
    image = cam.capture_array()
    #'''

    #image = "output_images/1.jpg" #"dataset/lane/lane_023.jpg"   # 141
 
    iamge = cam.capture_array()
    model = YOLO("object_detection/yolo_float16.tflite", task='detect')
    result = model(image, conf=0.5)[0]
    annotated_frame = result.plot()

    left_line, right_line = detect_lines(result)
    print("left:", left_line)
    print("right:", right_line)
    if left_line is not None and right_line is not None:
        midpoint = (left_line[0] + right_line[0]) / 2
        print("midpoint:", midpoint)
        cv2.circle(annotated_frame, (int(midpoint), 360), 5, (0, 0, 255), -1)
    #diff = left_line[0] / 480 
    #print(15 * float(diff))
    

    #annotated_frame = result.plot()
    cv2.imshow("camera", annotated_frame)
    cv2.waitKey(0)

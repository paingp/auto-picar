# Auto PiCar-X — Autonomous Lane-Following Robot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red?logo=raspberrypi)
![Framework](https://img.shields.io/badge/Model-YOLOv11-purple)
![License](https://img.shields.io/badge/License-MIT-green)

An autonomous robot car built on a **SunFounder PiCar-X** and powered by a **Raspberry Pi**. The car drives itself within lane marked by black tape and obeys toy road signs — all using real-time computer vision and a custom-trained deep learning model running entirely on-device.

---

## Overview

This project combines classical computer vision and modern deep learning to build a self-driving robot at the edge. The car navigates a track defined by black tape lane lines and responds to three road sign types — stop, caution, and crosswalk — using a YOLO11 model fine-tuned on a custom dataset and deployed as a quantized TFLite model optimized for the Raspberry Pi's hardware constraints.

The system runs three concurrent threads — frame capture, drive control, and obstacle avoidance — enabling real-time, responsive behavior without dedicated GPU hardware.

---

## Features

- **Dual-mode lane detection** — YOLO11-based AI detection with an OpenCV (Canny edge + Hough transform) fallback
- **Road sign recognition** — Detects and responds to stop, caution, and crosswalk signs in real time
- **Edge AI deployment** — YOLOv11 Nano model quantized to TFLite float16 (~5.4 MB) for on-device inference
- **Multi-threaded architecture** — Separate threads for frame capture, drive logic, and ultrasonic obstacle detection
- **Proportional steering control** — Gain-scaled offset from lane center with deadzone and jitter suppression
- **Obstacle avoidance** — Ultrasonic sensor halts the car when an obstacle is detected within 20 cm
- **Cliff detection** — Grayscale sensors prevent the car from driving off elevated surfaces
- **Manual drive mode** — Keyboard-controlled driving over SSH for testing and debugging
- **Modular functional tests** — Isolated test scripts for each subsystem (lane following, sign detection, obstacle avoidance, etc.)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Raspberry Pi                        │
│                                                          │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  Picamera2   │───▶│      capture_thread           │   │
│  │  640x480 RGB │    │  (frame buffer w/ condition   │   │
│  └──────────────┘    │   variable synchronization)  │   │
│                      └──────────────┬───────────────┘   │
│                                     │ frame ready        │
│  ┌──────────────┐    ┌──────────────▼───────────────┐   │
│  │  Ultrasonic  │    │       drive_thread            │   │
│  │   Sensor     │    │                               │   │
│  └──────┬───────┘    │  ┌─────────────────────────┐ │   │
│         │            │  │  YOLO11 Inference        │ │   │
│  ┌──────▼───────┐    │  │  (yolo_float16.tflite)   │ │   │
│  │  avoid_obs   │    │  └────────┬────────┬────────┘ │   │
│  │   thread     │    │           │        │           │   │
│  │  (stops car  │    │  Lane     │  Sign  │           │   │
│  │  if < 20cm)  │    │  Lines    │  Detect│           │   │
│  └──────────────┘    │     │         │               │   │
│                      │  ┌──▼─────────▼────────────┐ │   │
│                      │  │  Steering Control        │ │   │
│                      │  │  (offset → servo angle)  │ │   │
│                      │  └──────────┬───────────────┘ │   │
│                      └────────────┼───────────────────┘  │
│                                   │                       │
│                      ┌────────────▼───────────────┐      │
│                      │     PiCar-X Motors          │      │
│                      │   (drive + steering servo)  │      │
│                      └─────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| **Robot Platform** | SunFounder PiCar-X |
| **Compute** | Raspberry Pi 5 (recommended) |
| **Camera** | Raspberry Pi Camera Module (via Picamera2) |
| **Distance Sensor** | Ultrasonic sensor (built into PiCar-X) |
| **Ground Sensor** | Grayscale sensor array (built into PiCar-X) |
| **Track** | Black electrical tape on a light-colored surface |
| **Signs** | Toy road signs (stop, caution/yield, crosswalk) |

---

## Software Requirements

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLO11 model inference |
| `opencv-python` | Image processing, lane detection |
| `numpy` | Numerical computation |
| `picamera2` | Raspberry Pi camera interface |
| `picarx` | SunFounder PiCar-X motor/servo control |
| `robot_hat` | Hardware abstraction (TTS, sensors) |
| `sshkeyboard` | Keyboard input for manual drive mode |
| `psutil` | System resource monitoring |
| `tflite-runtime` | TFLite model inference on Raspberry Pi |

---

## How It Works

### Lane Detection

The system uses two complementary approaches:

**YOLO11 (Primary)**
The fine-tuned YOLO11 Nano model detects black lane line segments as a dedicated class. Detected bounding boxes are filtered to the lower half of the frame, sorted by vertical position, and assigned to left/right lanes based on their horizontal center. This approach is robust to varying lighting and partial occlusion.

**OpenCV (Fallback)**
A traditional pipeline provides a reliable fallback:
1. Grayscale conversion → Gaussian blur → binary threshold
2. Adaptive Canny edge detection (threshold derived from image median)
3. Trapezoidal region-of-interest mask (bottom 50% of frame)
4. Probabilistic Hough Line Transform to extract line segments
5. Lines filtered by slope magnitude (|slope| > 0.5) and split into left/right

### Steering Control

```
lane_midpoint = (left_lane_x + right_lane_x) / 2
offset        = lane_midpoint - frame_center_x     # pixels
norm_offset   = offset / 240                       # normalize to [-1, 1]
steering      = norm_offset * 24                   # scale to ±24° servo range

# Deadzone: ignore small offsets to prevent oscillation
if |offset| < 48px:  steering = 0

# Jitter suppression: ignore sub-degree angle changes
if |steering - prev_steering| < 1°:  steering = prev_steering
```

### Road Sign Recognition

The YOLO11 model detects three sign classes in every frame. Responses are triggered once a sign's bounding box height exceeds a threshold (indicating the car is close enough):

| Sign | Trigger (bbox height) | Response |
|------|----------------------|----------|
| Stop | > 10% of frame height | Full stop (permanent) |
| Crosswalk | > 10% of frame height | Stop briefly, then continue at reduced speed |
| Caution | > 80% of frame height | Reduce speed to 5 |

### Obstacle Avoidance

A dedicated background thread continuously polls the ultrasonic sensor. If a reading falls below 20 cm, the car stops immediately and the main drive loop exits.

---

## Model Training

The object detection model was trained using the **YOLO11 Nano** architecture on a custom road signs dataset.

| Detail | Value |
|--------|-------|
| Base model | `yolo11n.pt` (Nano — optimized for edge devices) |
| Dataset | Road Signs v5i (custom-labeled) |
| Classes | `caution`, `crosswalk`, `lane`, `stop` |
| Training epochs | 5 |
| Image size | 640×640 |
| Training environment | Google Colab |
| Export format | TFLite float16 |
| Model size | ~5.4 MB |

**Validation Results**

| Class | Precision | mAP50 | mAP50-95 |
|-------|-----------|-------|----------|
| Caution | 1.000 | 0.995 | — |
| Crosswalk | 0.999 | 0.995 | — |
| Lane | 0.992 | 0.995 | — |
| Stop | 0.997 | 0.995 | — |
| **Overall** | **0.997** | **0.995** | **0.803** |

The training notebook (`object_detection/finetune_yolo.ipynb`) documents the full pipeline from dataset preparation through model export.

---

## Project Structure

```
auto-picar/
├── auto_drive.py               # Main autonomous driving program
├── manual_drive.py             # SSH keyboard-controlled manual drive
├── ai_lane_detection.py        # YOLO11-based lane line detection
├── opencv_lane_detection.py    # OpenCV Canny/Hough lane detection
│
├── functional_testing/
│   ├── lane_following.py       # Lane detection + steering test (no ML)
│   ├── follow_signs.py         # Road sign detection + response test
│   ├── detect_objects.py       # YOLO inference benchmarking
│   ├── avoid_obstacles.py      # Ultrasonic obstacle avoidance test
│   ├── cliff_detection.py      # Grayscale cliff/edge detection test
│   └── camera.py               # Basic camera capture test
│
└── object_detection/
    ├── finetune_yolo.ipynb     # YOLO11 training notebook (Google Colab)
    ├── collect_images.py       # Dataset image collection utility
    ├── inference.py            # Model inference debugging script
    └── yolo_float16.tflite     # Compiled TFLite model (~5.4 MB)
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/auto-picar.git
cd auto-picar
```

### 2. Install system dependencies (on Raspberry Pi)

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libcamera-dev
```

### 3. Install Python packages

```bash
pip install ultralytics opencv-python numpy picamera2 sshkeyboard psutil
# Install SunFounder libraries per the PiCar-X documentation
pip install robot-hat picarx
```

> **Note:** `tflite-runtime` may require a Raspberry Pi-specific wheel. Refer to the [TFLite installation guide](https://www.tensorflow.org/lite/guide/python) for your Pi OS version.

### 4. Enable the camera

```bash
sudo raspi-config
# Interface Options → Camera → Enable
```

### 5. Set up the track

- Lay two parallel strips of black electrical tape on a light-colored floor, spaced approximately 20–30 cm apart.
- Place toy road signs along the track.

---

## Usage

### Autonomous Drive

```bash
python auto_drive.py
```

Press `Ctrl+C` to stop.

### Manual Drive (via SSH keyboard)

```bash
python manual_drive.py
```

| Key | Action |
|-----|--------|
| `W` | Forward |
| `S` | Backward |
| `A` | Steer left |
| `D` | Steer right |
| Any other | Stop & exit |

### Functional Tests

Run individual subsystem tests from the `functional_testing/` directory:

```bash
# Test lane detection and steering only
python functional_testing/lane_following.py

# Test road sign detection and response
python functional_testing/follow_signs.py

# Benchmark YOLO inference on the Pi
python functional_testing/detect_objects.py

# Test ultrasonic obstacle avoidance
python functional_testing/avoid_obstacles.py

# Test cliff/edge detection
python functional_testing/cliff_detection.py
```

---

## Configuration

Key parameters in `auto_drive.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speed` | `10–25` | Motor drive speed (scale: –100 to 100) |
| `dthresh` | `20` cm | Obstacle stop distance |
| Confidence threshold | `0.5` | YOLO detection minimum confidence |
| Lane deadzone | `48` px | Minimum offset before steering correction |
| Max steering angle | `±24°` | Servo range limit |
| Steering gain | `0.5` | Offset-to-angle scaling factor |
| Camera resolution | `640×480` | Input frame size |
| Sign height threshold | `0.1–0.8` | Normalized bbox height to trigger sign action |

---

## Results

| Metric | Value |
|--------|-------|
| Sign detection mAP50 | 99.5% |
| Inference time (Pi) | ~4.8 ms/frame |
| Effective control rate | ~10 Hz |
| Model size (TFLite float16) | 5.4 MB |
| Steering range | ±24° |

---

## Future Improvements

- **PID steering controller** — Replace the current proportional controller with a full PID loop to reduce overshoot on sharp curves
- **Adaptive speed control** — Automatically vary speed based on curve sharpness
- **More sign classes** — Extend the model to recognize speed limit signs, yield signs, or traffic lights
- **ROS2 integration** — Port the architecture to ROS2 for better modularity and logging
- **Dataset expansion** — Collect more training data under varied lighting conditions to improve robustness
- **Onboard logging dashboard** — Stream telemetry (speed, steering angle, detected objects) to a web interface in real time

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

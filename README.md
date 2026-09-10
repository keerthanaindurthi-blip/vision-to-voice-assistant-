# Vision-to-Voice Guidance Assistant

A Python-based computer vision application that analyzes uploaded videos with YOLOv4-tiny, converts detected objects and their approximate horizontal positions into text descriptions, and provides optional text-to-speech playback.

> Current scope: the submitted implementation processes uploaded video files. It does not currently implement live camera input, OCR, scene description, or natural-language generation.

## Problem Statement

People with visual impairments may need accessible ways to receive information about objects in their surroundings. A computer-vision system can help by detecting recognizable objects and communicating the results through speech.

## Objective

The objective of this project is to demonstrate a vision-to-voice pipeline that accepts a video, detects recognizable objects and people, estimates their approximate horizontal position, presents the result as text, and optionally converts the result to speech.

## Key Features

- Video upload through a Streamlit interface.
- Object and person detection using YOLOv4-tiny through OpenCV DNN.
- Confidence thresholding and non-maximum suppression.
- Frame sampling to reduce repeated processing.
- Approximate horizontal positioning: left, ahead, or right.
- Duplicate suppression for repeated object/position descriptions.
- Processing progress indicator.
- Text output of detected objects.
- Optional text-to-speech playback with pyttsx3.
- Includes sample videos for testing.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web-based user interface |
| OpenCV | Video processing and YOLO inference |
| YOLOv4-tiny | Object detection model |
| NumPy | Numerical processing |
| pyttsx3 | Local text-to-speech |

## Model and Class Data

The project includes YOLOv4-tiny configuration and weights plus the COCO class-name file in `yolo/`. The supplied `yolo/README.txt` identifies the source repositories for these files.

## How the System Works

```text
Uploaded Video
      |
      v
OpenCV Video Reader
      |
      v
Sample Video Frames
      |
      v
YOLOv4-tiny Object Detection
      |
      v
Confidence Filtering + NMS
      |
      v
Estimate Horizontal Position
(left / ahead / right)
      |
      v
Remove Duplicate Descriptions
      |
      v
Text Result
      |
      v
Optional pyttsx3 Text-to-Speech
```

## Project Workflow

1. The user uploads a supported video through Streamlit.
2. The application loads YOLOv4-tiny configuration, weights, and COCO class names.
3. The video is sampled and processed with OpenCV.
4. YOLOv4-tiny detects objects in sampled frames.
5. Confidence filtering and non-maximum suppression reduce weak and overlapping detections.
6. Bounding-box centers are used to classify detections as left, ahead, or right.
7. Duplicate object/position descriptions are removed.
8. The resulting text is displayed and can be converted to speech with pyttsx3.

## Installation

### Prerequisites

- Python 3.10 or newer is recommended.
- A working local text-to-speech engine compatible with pyttsx3.
- The YOLO model files included in the `yolo/` directory.

### Setup

```bash
git clone https://github.com/keerthanaindurthi-blip/vision-to-voice-assistant-.git
cd vision-to-voice-assistant-
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run app.py
```

Open the local Streamlit address, upload a sample video or supported video file, and select **Start processing**. After processing, select **Convert to speech** for speech playback.

## Example Usage

For a video containing recognizable COCO objects, the application may produce a result such as:

```text
Person ahead of you. Car on your left.
```

The exact output depends on the video and model detections.

## Configuration

The current detection settings in `app.py` are:

- `CONFIDENCE_THRESHOLD = 0.5`
- `NMS_THRESHOLD = 0.4`
- `FRAME_SKIP = 10`

## Repository Structure

```text
vision-to-voice-assistant-/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── cars.mp4
├── motorbikes.mp4
├── ppe-2.mp4
└── yolo/
    ├── README.txt
    ├── coco.names
    ├── yolov4-tiny.cfg
    └── yolov4-tiny.weights
```

## Current Limitations

- Input is uploaded video rather than a live camera stream.
- Position is an approximate left/center/right classification.
- The system does not estimate object distance or depth.
- Detection is limited to classes in the supplied COCO names file.
- OCR is not implemented in the submitted code.
- No generative scene-description or text-generation model is implemented.
- Text-to-speech depends on the local machine's speech engine.
- No mobile or wearable hardware integration is included in this version.

## Future Enhancements

- Real-time camera-based detection.
- Distance/depth estimation.
- OCR for reading visible text.
- More contextual scene descriptions.
- Improved speech prioritization.
- Mobile or edge-device deployment.
- Additional accessibility controls.

## Author

**Keerthana Indurthi**  
Computer Science and Engineering Student

---

Educational and portfolio project demonstrating computer vision, video processing, and text-to-speech integration.

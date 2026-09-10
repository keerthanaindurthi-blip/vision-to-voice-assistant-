"""
Vision-to-Voice Assistant for Blind Users
=========================================
Simple flow: Upload video -> Detect objects and persons -> Show result as text -> Convert to speech.
"""

import streamlit as st
import cv2
import numpy as np
import pyttsx3
import tempfile
from pathlib import Path
from threading import Thread

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
YOLO_DIR = Path("yolo")
FRAME_SKIP = 10


def get_yolo_paths():
    cfg = YOLO_DIR / "yolov4-tiny.cfg"
    weights = YOLO_DIR / "yolov4-tiny.weights"
    names = YOLO_DIR / "coco.names"
    return cfg, weights, names


def load_coco_names(names_path):
    if not names_path.exists():
        return []
    with open(names_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_yolo_net(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def get_position(x_center, frame_w):
    third = frame_w / 3
    if x_center < third:
        return "on your left"
    if x_center > 2 * third:
        return "on your right"
    return "ahead of you"


def process_frame(net, output_layers, frame, class_names, conf_thresh, nms_thresh):
    """Run YOLO on one frame; return (class_id, confidence, box) results."""
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    h, w = frame.shape[:2]
    scale_x, scale_y = w / 416.0, h / 416.0
    boxes, confidences, class_ids = [], [], []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < conf_thresh:
                continue
            cx = int(detection[0] * scale_x)
            cy = int(detection[1] * scale_y)
            bw = int(detection[2] * scale_x)
            bh = int(detection[3] * scale_y)
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            boxes.append([x1, y1, bw, bh])
            confidences.append(conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((class_ids[i], confidences[i], boxes[i]))
    return result


def process_video_to_text(video_path, net, output_layers, class_names, progress_bar=None):
    """Process sampled video frames and return a single text description."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    seen = set()
    lines = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                if progress_bar and total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))
                continue

            detections = process_frame(
                net, output_layers, frame, class_names,
                CONFIDENCE_THRESHOLD, NMS_THRESHOLD
            )

            for class_id, _conf, (x, _y, bw, _bh) in detections:
                name = class_names[class_id] if class_id < len(class_names) else "object"
                x_center = x + bw / 2
                position = get_position(x_center, width)
                key = (name.lower(), position)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{name.capitalize()} {position}.")

            frame_idx += 1
            if progress_bar and total_frames > 0:
                progress_bar.progress(min(1.0, frame_idx / total_frames))
    finally:
        cap.release()

    if not lines:
        return "No objects or persons detected in the video."
    return " ".join(lines)


def speak_text(text):
    """Run local text-to-speech in a background thread."""
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    Thread(target=_run, daemon=True).start()


def check_yolo_files():
    cfg, weights, names = get_yolo_paths()
    if not cfg.exists():
        return False, f"Missing {cfg}. Add yolov4-tiny.cfg to the 'yolo' folder."
    if not weights.exists():
        return False, f"Missing {weights}. Add yolov4-tiny.weights to the 'yolo' folder."
    if not names.exists():
        return False, f"Missing {names}. Add coco.names to the 'yolo' folder."
    return True, "OK"


def main():
    st.set_page_config(
        page_title="Vision-to-Voice Assistant for Blind Users",
        layout="wide",
    )
    st.title("Vision-to-Voice Guidance Assistant")
    st.markdown(
        "**Upload a video** -> detect objects and persons -> review the result as text -> convert the result to speech."
    )

    ok, msg = check_yolo_files()
    if not ok:
        st.error(msg)
        return

    if "result_text" not in st.session_state:
        st.session_state.result_text = None

    uploaded = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        help="MP4, AVI, MOV, MKV",
    )

    if uploaded is None:
        st.info("Upload a video to begin.")
        return

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded.name).suffix
    ) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name

    @st.cache_resource
    def load_model():
        cfg, weights, names_path = get_yolo_paths()
        class_names = load_coco_names(names_path)
        net = load_yolo_net(cfg, weights)
        output_layers = get_output_layers(net)
        return net, output_layers, class_names

    try:
        net, output_layers, class_names = load_model()
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return

    if st.button("Start processing", type="primary"):
        progress_bar = st.progress(0.0)
        status = st.empty()
        status.write("Detecting objects and persons...")

        try:
            result_text = process_video_to_text(
                video_path, net, output_layers, class_names, progress_bar=progress_bar
            )
            st.session_state.result_text = result_text
            status.success("Done. See result below.")
        except Exception as e:
            st.error(f"Processing error: {e}")
            status.empty()
        progress_bar.empty()

    if st.session_state.result_text is not None:
        st.subheader("Result (text)")
        st.text_area(
            "Detected objects and persons",
            st.session_state.result_text,
            height=120,
            disabled=True,
        )

        if st.button("Convert to speech", type="secondary"):
            speak_text(st.session_state.result_text)
            st.caption("Speech playback started. Check your system volume.")


if __name__ == "__main__":
    main()

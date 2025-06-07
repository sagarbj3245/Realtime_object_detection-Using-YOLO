import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import torch
import pyttsx3
from collections import Counter
import threading
import time

# Load YOLOv8 model with GPU support if available
MODEL = "yolov8x.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL).to(device)
model.fuse()

# Dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# Unique class IDs to track in the video stream
selected_classes = list(set([0, 2, 3, 5, 23, 25, 39, 44, 56, 63, 64, 66, 76]))

# Create instance of BoxAnnotator for drawing bounding boxes
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

def speak_summary(counts):
    """
    Generate and speak out the summary of detected objects.
    """
    summary = ", ".join(
        f"{count} {CLASS_NAMES_DICT[class_id]}{'s' if count > 1 else ''} detected"
        for class_id, count in counts.items()
    )
    engine.say(summary)
    engine.runAndWait()

def audio_thread(new_detections, stop_event):
    """
    Thread function to periodically speak out detected objects.
    """
    while not stop_event.is_set():
        if new_detections:
            speak_summary(new_detections)
            new_detections.clear()
        time.sleep(1)

# Open webcam for video capture
cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Adjust frame size for better performance
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Start the audio thread
new_detections = Counter()
stop_event = threading.Event()
thread = threading.Thread(target=audio_thread, args=(new_detections, stop_event))
thread.daemon = True
thread.start()

# To keep track of previous counts
prev_counts = Counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    try:
        results = model(small_frame, verbose=False)[0]

        # Manually create Detections object
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

        # Scale detections back to original frame size
        detections.xyxy *= 2

        # Filter to only selected classes
        detections = detections[np.isin(detections.class_id, selected_classes)]

        # Count current objects
        current_counts = Counter(detections.class_id)

        # Detect new or increased counts
        for class_id, count in current_counts.items():
            if count > prev_counts.get(class_id, 0):
                new_detections[class_id] += count - prev_counts.get(class_id, 0)

        prev_counts = current_counts.copy()

        # Prepare labels for drawing
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {current_counts[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Draw boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    except Exception as e:
        print(f"Error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean exit
cap.release()
cv2.destroyAllWindows()
stop_event.set()

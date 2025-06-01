import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

model = YOLO("yolov8m.pt")
outline_img = cv2.imread("assets/person_outline.png", cv2.IMREAD_UNCHANGED)

if outline_img is None:
    raise FileNotFoundError("person_outline.png not found.")

os.makedirs("captured_images", exist_ok=True)

def is_person_inside_box_75_percent(person_box, guide_box):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = guide_box
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    person_area = (px2 - px1) * (py2 - py1)
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    return intersection_area / person_area >= 0.80

def extract_outline_contour(image, thickness=3):
    if image.shape[2] == 4:
        image = image[..., :3]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness)
    return contour_img

def start_capture():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    countdown_started = False
    countdown_start_time = None
    photo_captured = False

    while cap.isOpened() and not photo_captured:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        outline_height = int(frame_height * 0.95)
        outline_width = int(frame_width * 0.5)
        outline_resized = cv2.resize(outline_img, (outline_width, outline_height))
        contour_mask = extract_outline_contour(outline_resized, thickness=3)

        x_offset = (frame_width - outline_width) // 2
        y_offset = (frame_height - outline_height) // 2
        outline_box = (x_offset, y_offset, x_offset + outline_width, y_offset + outline_height)

        results = model(frame)[0]
        person_boxes = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            if model.names[class_id] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                person_boxes.append(((x1, y1, x2, y2), area))

        if person_boxes:
            closest_box, _ = max(person_boxes, key=lambda b: b[1])
            height, width = frame.shape[:2]
            guide_margin = 80
            guide_box = (guide_margin, guide_margin, width - guide_margin, height - guide_margin)

            if is_person_inside_box_75_percent(closest_box, guide_box):
                if not countdown_started:
                    countdown_started = True
                    countdown_start_time = datetime.now()
                else:
                    elapsed = (datetime.now() - countdown_start_time).total_seconds()
                    if elapsed >= 5:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"captured_images/person_{timestamp}.jpg"
                        cv2.imwrite(filename, clean_frame)
                        photo_captured = True
                        cap.release()
                        return filename
            else:
                countdown_started = False

    cap.release()
    return None

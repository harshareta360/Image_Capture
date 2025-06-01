import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Load outline image (must have alpha channel)
outline_img = cv2.imread("assets/person_outline.png", cv2.IMREAD_UNCHANGED)

if outline_img is None:
    raise FileNotFoundError("person_outline.png not found. Make sure it's in the working directory.")

# Create folder to save captured photos
os.makedirs("captured_images", exist_ok=True)

def is_person_inside_box_80_percent(person_box, guide_box):
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
    """Extract only the white silhouette contour from outline image."""
    if image.shape[2] == 4:
        image = image[..., :3]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness)
    return contour_img

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

countdown_started = False
countdown_start_time = None
photo_captured = False
message = ""

while cap.isOpened() and not photo_captured:
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    # Resize outline image and extract its contour
    outline_height = int(frame_height * 0.95)
    outline_width = int(frame_width * 0.5)
    outline_resized = cv2.resize(outline_img, (outline_width, outline_height))
    contour_mask = extract_outline_contour(outline_resized, thickness=3)

    # Convert to BGR so we can color the border green instead of white
    green_contour = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)
    green_contour[np.where((green_contour == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

    # Position to center overlay
    x_offset = (frame_width - outline_width) // 2
    y_offset = (frame_height - outline_height) // 2
    x1_box, y1_box = x_offset, y_offset
    x2_box, y2_box = x_offset + outline_width, y_offset + outline_height
    outline_box = (x1_box, y1_box, x2_box, y2_box)

    # Draw the green contour overlay
    for y in range(outline_height):
        for x in range(outline_width):
            if contour_mask[y, x] > 0:
                frame[y_offset + y, x_offset + x] = green_contour[y, x]

    # YOLOv8 inference
    results = model(frame)[0]
    person_boxes = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        if model.names[class_id] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            person_boxes.append(((x1, y1, x2, y2), area))

    if not person_boxes:
        message = "No person detected. Please stand in front of the camera."
        countdown_started = False
    else:
        closest_box, _ = max(person_boxes, key=lambda b: b[1])
        x1, y1, x2, y2 = closest_box

        height, width = frame.shape[:2]
        guide_margin = 80
        gx1, gy1 = guide_margin, guide_margin
        gx2, gy2 = width - guide_margin, height - guide_margin
        guide_box = (gx1, gy1, gx2, gy2)

        if is_person_inside_box_80_percent(closest_box, guide_box):
            if not countdown_started:
                countdown_started = True
                countdown_start_time = datetime.now()
                message = "⏳ Hold still! Capturing in 3 seconds..."
            else:
                elapsed = (datetime.now() - countdown_start_time).total_seconds()
                if elapsed >= 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_images/person_{timestamp}.jpg"
                    cv2.imwrite(filename, clean_frame)
                    message = "✅ Photo captured successfully!"
                    photo_captured = True
                else:
                    message = f"Hold still! Capturing in {5 - int(elapsed)}"
        else:
            countdown_started = False
            message = "fit inside the outline."

    # Display status message
    (text_width, text_height), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (30, 20), (30 + text_width + 20, 20 + text_height + 20), (0, 0, 0), -1)
    cv2.putText(frame, message, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Smart Auto-Capture Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

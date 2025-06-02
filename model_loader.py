# # model_loader.py
# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load YOLO model once
# model = YOLO("yolov8m.pt")

# # Load and resize the outline image (alpha channel assumed)
# outline_img = cv2.imread("assets/person_outline.png", cv2.IMREAD_UNCHANGED)

# def extract_outline_contour(image, thickness=3):
#     if image.shape[2] == 4:
#         image = image[..., :3]
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour_img = np.zeros_like(gray)
#     cv2.drawContours(contour_img, contours, -1, (255), thickness)
#     return contour_img

# def is_person_inside_box_85_percent(person_box, guide_box):
#     px1, py1, px2, py2 = person_box
#     gx1, gy1, gx2, gy2 = guide_box
#     ix1 = max(px1, gx1)
#     iy1 = max(py1, gy1)
#     ix2 = min(px2, gx2)
#     iy2 = min(py2, gy2)
#     if ix1 >= ix2 or iy1 >= iy2:
#         return False
#     person_area = (px2 - px1) * (py2 - py1)
#     intersection_area = (ix2 - ix1) * (iy2 - iy1)
#     return intersection_area / person_area >= 0.85
import cv2
import numpy as np
from ultralytics import YOLO

model = None
outline_img = None

def load_model():
    global model
    if model is None:
        model = YOLO("yolov8m.pt")
    return model

def load_outline_image():
    global outline_img
    if outline_img is None:
        outline_img = cv2.imread("assets/person_outline.png", cv2.IMREAD_UNCHANGED)
    return outline_img

def extract_outline_contour(image, thickness=3):
    if image is None:
        raise ValueError("Outline image not found or failed to load.")
    if image.shape[2] == 4:
        image = image[..., :3]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness)
    return contour_img

def is_person_inside_box_85_percent(person_box, guide_box):
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
    return intersection_area / person_area >= 0.85

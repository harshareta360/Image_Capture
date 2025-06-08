import cv2
import numpy as np
import os

def extract_and_save_outline(image_path, output_path="outline_mask.png", outline_thickness=2):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate black figure
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create transparent image with 4 channels (RGBA)
    h, w = gray.shape
    outline_img = np.zeros((h, w, 4), dtype=np.uint8)

    # Draw white outline with alpha
    cv2.drawContours(outline_img, contours, -1, (255, 255, 255, 255), thickness=outline_thickness)

    # Save image with transparency (PNG)
    cv2.imwrite(output_path, outline_img)
    print(f"[✔] Outline saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    extract_and_save_outline("assets/person_outline.jpeg")

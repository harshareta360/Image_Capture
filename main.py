from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO
from model_loader import model, outline_img, extract_outline_contour, is_person_inside_box_85_percent

app = FastAPI()

@app.post("/detect-capture")
async def detect_and_capture(files: list[UploadFile] = File(...)):
    for upload_file in files:
        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        clean_frame = frame.copy()

        frame_height, frame_width = frame.shape[:2]
        outline_height = int(frame_height * 0.95)
        outline_width = int(frame_width * 0.5)
        outline_resized = cv2.resize(outline_img, (outline_width, outline_height))
        contour_mask = extract_outline_contour(outline_resized)

        x_offset = (frame_width - outline_width) // 2
        y_offset = (frame_height - outline_height) // 2
        guide_box = (x_offset, y_offset, x_offset + outline_width, y_offset + outline_height)

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
            if is_person_inside_box_85_percent(closest_box, guide_box):
                # Convert the captured image to bytes
                _, buffer = cv2.imencode('.jpg', clean_frame)
                return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

    return {"message": "No valid person found in any frame."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


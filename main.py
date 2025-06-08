from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import logging
import base64 # Added for base64 encoding/decoding

from model_loader import process_image_from_frontend # Use the high-level wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Replace with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Smart Capture API is Live!"}

@app.post("/detect-capture")
async def detect_and_capture(files: list[UploadFile] = File(...)):
    logger.info(f"Received request to /detect-capture with {len(files)} files")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    upload_file = files[0] # Assuming only one file per request for simplicity
    logger.info(f"Processing file: {upload_file.filename}")

    contents = await upload_file.read()
    base64_original_image = base64.b64encode(contents).decode('utf-8')
    base64_original_image_with_prefix = f"data:{upload_file.content_type};base64,{base64_original_image}"

    result = process_image_from_frontend(base64_original_image_with_prefix)

    if "error" in result:
        logger.error(f"Error processing image: {result['error']}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")

    logger.info(f"Detection result: Status={result['status']}, Message={result.get('message', 'N/A')}, Score={result.get('positioning_score', 'N/A')}")

    # Only send status and message, no frame for detect-capture to reduce lag
    return {
        "status": result["status"],
        "message": result["message"],
        # "frame": result['frame'] # Removed to prevent frontend lag
    }

@app.post("/capture-final")
async def capture_final(files: list[UploadFile] = File(...)):
    logger.info(f"Received request to /capture-final with {len(files)} files")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    upload_file = files[0]
    logger.info(f"Processing file: {upload_file.filename}")

    contents = await upload_file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    base64_image_with_prefix = f"data:{upload_file.content_type};base64,{base64_image}"

    result = process_image_from_frontend(base64_image_with_prefix)

    if "error" in result:
        logger.error(f"Error processing image: {result['error']}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")

    logger.info(f"Detection result: Status={result['status']}, Message={result.get('message', 'N/A')}, Score={result.get('positioning_score', 'N/A')}")

    # For capture-final, always return the annotated image for feedback, as it's a final capture attempt
    return {
        "status": result["status"],
        "message": result["message"],
        "frame": result['frame'] # This is already base64 with prefix
    }

# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting server on http://127.0.0.1:8000")
#     uvicorn.run("main:app", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

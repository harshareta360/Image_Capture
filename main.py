from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from capture_logic import start_capture

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Smart Auto-Capture Camera API running."}

@app.get("/capture")
def capture_photo():
    filepath = start_capture()
    if filepath:
        return JSONResponse({"status": "success", "file": filepath})
    return JSONResponse({"status": "error", "message": "Failed to capture photo."})

import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import pytesseract
import re
from pydantic import BaseModel
import uvicorn

def read_img(img):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    text = pytesseract.image_to_string
    return (text)

app = FastAPI(debug=True)

class ImageType(BaseModel):
    url: str
    
    
@app.get('/')
def home():
    return {"Welcome ": "User"}

@app.post("/predict/")
def prediction(request: Request, file: bytes = File(...)):
    if request.method == "POST":
        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype = np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)
        return label
        return "No post request found."
         
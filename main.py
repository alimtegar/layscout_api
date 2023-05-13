import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from utils import get_image_urls, load_images_from_urls, resize_images, calc_map

MODEL_DIR = './model_weights/web_component_detection_model.pt'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      'http://localhost',
      'http://localhost:5173',
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(MODEL_DIR)
model.fuse()

class SearchParams(BaseModel):
  q: str
  layout_width: int
  layout: List[List[int]]
  limit: int = 10
  iou_threshold: float = 0.5

@app.get('/')
def root():
  return {'message': 'Hello, world!'}

@app.post('/search')
def search(params: SearchParams): 
  # Check if keywords is empty
  if not params.q:
    raise HTTPException(status_code=400, detail="Keywords cannot be empty.")
  
  # Check if layout array is empty
  if not params.layout:
    raise HTTPException(status_code=400, detail="Layout array cannot be empty.")
  
  np_layout = np.array(params.layout)
  
  # Check the shape of the layout array
  if np_layout.shape[1] != 6:
    raise HTTPException(status_code=400, detail="Layout array must have 6 columns.")
   
  image_urls = get_image_urls(keywords=params.q, limit=params.limit)
  images = load_images_from_urls(image_urls)
  images = resize_images(images=images, width=params.layout_width)
  
  detections = model(images)
  maps = [
    calc_map(
      detections=detection.boxes.data.numpy(), 
      ground_truth=np_layout,
      iou_threshold=params.iou_threshold,
    ) 
    for detection 
    in detections
  ]
  
  results = sorted([{
    'image_url': image_url,
    'map': map,
    'boxes': detection.boxes.data.tolist(), 
  } 
  for image_url, detection, map in zip(image_urls, detections, maps)
  if map > 0], key=lambda x: x['map'], reverse=True)
  
  return {'results': results}
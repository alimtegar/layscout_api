import numpy as np
import asyncio
import aioredis
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from utils import get_image_urls, load_images_from_urls, resize_images, calc_map, store_list_in_redis, get_list_from_redis

MODEL_DIR = './model_weights/web_component_detection_model.pt'
REDIS_KEY_PREFIX = 'layscout_'

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

redis = None

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

@app.on_event('startup')
async def startup_event():
  global redis
  redis = await aioredis.create_redis_pool('redis://localhost:6379')

@app.on_event('shutdown')
async def shutdown_event():
  redis.close()
  await redis.wait_closed()

@app.post('/search')
async def search(params: SearchParams): 
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
  
  # If the image is already detected, get the detection boxes from the cache
  cached_detectionx_boxes_list = []
  cached_image_urls = []
  uncached_image_urls = []
  
  for image_url in image_urls:
    redis_key = f'{REDIS_KEY_PREFIX}{image_url}'.encode()
    if redis_key in await redis.keys(f'{REDIS_KEY_PREFIX}*'):
      cached_detectionx_boxes = await get_list_from_redis(redis=redis, key=redis_key)
      if cached_detectionx_boxes:
        cached_detectionx_boxes_list.append(cached_detectionx_boxes)
        cached_image_urls.append(image_url)
      else:
        uncached_image_urls.append(image_url)
    else:
      uncached_image_urls.append(image_url)
      
  uncached_detection_boxes_list = []
  if uncached_image_urls:
    images = load_images_from_urls(uncached_image_urls)
    images = resize_images(images=images, width=params.layout_width)
    
    detections = model(images)
    uncached_detection_boxes_list = [detection.boxes.data.tolist() for detection in detections]
  
  all_image_urls = cached_image_urls + uncached_image_urls
  all_detection_boxes_list = cached_detectionx_boxes_list + uncached_detection_boxes_list
  
  maps = [
    calc_map(
      detections=np.array(detection_boxes), 
      ground_truth=np_layout,
      iou_threshold=params.iou_threshold,
    ) 
    for detection_boxes in all_detection_boxes_list
  ]
  
  results = []
  for image_url, detection_boxes, map in zip(all_image_urls, all_detection_boxes_list, maps):
    # Cache the detection boxes
    await store_list_in_redis(redis, key=f'{REDIS_KEY_PREFIX}{image_url}', lst=detection_boxes)
    
    if map > 0:
      results.append({
        'image_url': image_url,
        'map': map,
        'boxes': detection_boxes, 
      })
  results = sorted(results, key=lambda x: x['map'], reverse=True)
  
  return {'results': results}
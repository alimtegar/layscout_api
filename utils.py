import numpy as np
import cv2
import requests
from typing import List, Tuple
from simple_image_download import simple_image_download as simp

def load_image_from_url(url: str) -> np.ndarray:
    response = requests.get(url)

    if response.status_code != 200:
        return np.empty((0,))

    image_content = response.content
    image = np.asarray(bytearray(image_content), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_images_from_urls(urls: List[str]) -> List[np.ndarray]:
  images = []
  for i, url in enumerate(urls):
    image = load_image_from_url(url)
    if image.size:
      images.append(image)
  return images

def get_image_urls(keywords: str, limit: int) -> List[str]:
  response = simp.simple_image_download()
  urls = response.urls(keywords=keywords, limit=3+limit)
  urls = [url for url in urls if 'www.gstatic.com' not in url]
  return urls

def resize_images(images: List[np.ndarray], width: int) -> List[np.ndarray]:
  resized_images = []
  for image in images:
    height, current_width, _ = image.shape
    scale = width / current_width
    new_height = int(height * scale)
    new_size = (width, new_height)
    resized_image = cv2.resize(image, new_size)
    resized_images.append(resized_image)
  return resized_images

def calc_iou(box1: np.ndarray, box2: np.ndarray) -> float:
  # Calculate intersection coordinates
  xmin = max(box1[0], box2[0])
  ymin = max(box1[1], box2[1])
  xmax = min(box1[2], box2[2])
  ymax = min(box1[3], box2[3])
  # Calculate intersection area
  iw = max(0.0, xmax - xmin)
  ih = max(0.0, ymax - ymin)
  inters = iw * ih
  # Calculate union area
  union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
          (box2[2] - box2[0]) * (box2[3] - box2[1]) - \
          inters
  # Calculate IoU
  iou = inters / union if union > 0 else 0.0
  return iou

def calc_precision_recall(detections: np.ndarray, ground_truth: np.ndarray, iou_threshold: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
  tp = np.zeros(len(detections))
  fp = np.zeros(len(detections))
  fn = np.zeros(len(ground_truth))
  for i, det in enumerate(detections):
    ious = [calc_iou(det[:4], gt[:4]) for gt in ground_truth]
    max_iou = max(ious) if ious else 0
    if max_iou >= iou_threshold and det[5] == ground_truth[ious.index(max_iou)][5]:
      tp[i] = 1
    else:
      fp[i] = 1
  for i, gt in enumerate(ground_truth):
    ious = [calc_iou(gt[:4], det[:4]) for det in detections]
    if max(ious) < iou_threshold:
      fn[i] = 1
  tp = np.cumsum(tp)
  fp = np.cumsum(fp)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  return precision, recall

def calc_ap(precision: np.ndarray, recall: np.ndarray) -> float:
  recall = np.concatenate(([0], recall, [1]))
  precision = np.concatenate(([0], precision, [0]))
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = max(precision[i], precision[i+1])
  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  ap = np.sum((recall[indices] - recall[indices-1]) * precision[indices])
  return ap

def calc_map(detections: np.ndarray, ground_truth: np.ndarray, iou_threshold: float=0.5) -> float:
  print('iou_threshold = ', iou_threshold)
  class_ids = np.unique(np.concatenate([detections[:,5], ground_truth[:,5]]))
  aps = []
  for cls in class_ids:
    cls_detections = detections[detections[:,5] == cls]
    cls_ground_truth = ground_truth[ground_truth[:,5] == cls]

    # Resize detection and ground truth arrays to have the same length,
    # if necessary, so that they can be compared and used to calculate
    # precision and recall.
    num_detections = len(cls_detections)
    num_ground_truth = len(cls_ground_truth)
    if num_detections > num_ground_truth:
      cls_ground_truth.resize(*cls_detections.shape)
    elif num_detections < num_ground_truth:
      cls_detections.resize(*cls_ground_truth.shape)

    precision, recall = calc_precision_recall(cls_detections, cls_ground_truth, iou_threshold)
    ap = calc_ap(precision, recall)
    aps.append(ap)
  return np.mean(aps)
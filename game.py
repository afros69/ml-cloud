from typing import Any

import PIL
import cv2
import torch
from ultralytics import YOLO
from colors_ml import device, data_transforms_val, padding, best_model_wts
from consts import number_map, detection_model_path, color_map
import supervision as sv


class Game:
    game_sid: str
    color_cache: dict
    detection: Any
    store_frames: bool
    frames: list

    def __init__(self, game_sid: str, store_frames: bool = False):
        self.game_sid = game_sid
        self.color_cache = {}
        self.load_model()
        self.store_frames = store_frames
        self.frames = []

    def load_model(self):
        self.detection = YOLO(detection_model_path)
        print("model loaded")

    def reset(self):
        self.color_cache = {}
        self.load_model()

    def predict_color(self, img, x1, y1, x2, y2):
        crop = img[y1:y2, x1:x2].copy()
        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = padding(image=crop)['image']
        crop = PIL.Image.fromarray(crop)
        crop = data_transforms_val(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = best_model_wts(crop)
            predicted = torch.argmax(y_pred, axis=1)
            color = color_map.get(predicted.item(), "?")
        return color

    def handle_frame(self, frame):
        results = self.detection.track(frame, classes=[0], persist=True, conf=0.6)
        # detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.5, class_agnostic=False)
        if len(results) == 0:
            return None

        result_boxes = []

        result = results[0]
        boxes = result.boxes
        if len(result.boxes) == 0:
            return None

        names = result.names
        ids = result.boxes.id
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box_number = 1
        for box, alt_box, cls, conf, id in zip(boxes.data, boxes.xywh.data, boxes.cls, boxes.conf, ids.numpy()):
            x1, y1, x2, y2 = box[:4].int().tolist()
            x, y, w, h = alt_box[:4].int().tolist()
            stored = id in self.color_cache
            color = self.color_cache.get(id, "?")
            if names[int(cls)] == 'ball' and not stored:
                color = self.predict_color(img, x1, y1, x2, y2)
            if color.endswith("_str"):
                self.color_cache[id] = color
            box_number += 1
            result_boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "color": color,
                "number": number_map.get(color, 0)
            })

        return result_boxes


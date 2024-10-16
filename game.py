from typing import Any
import heapq
from collections import defaultdict

import PIL
import cv2
import torch
from ultralytics import YOLO
from colors_ml import device, data_transforms_val, padding, best_model_wts
from consts import number_map, detection_model_path, color_map
import supervision as sv


class Game:
    game_sid: str
    detection: Any
    store_frames: bool
    frames: list

    def __init__(self, game_sid: str, store_frames: bool = False):
        self.game_sid = game_sid
        self.load_model()
        self.store_frames = store_frames
        self.frames = []

        self.color_map = defaultdict(lambda: defaultdict(int))
        self.max_heap_map = defaultdict(list)

    def load_model(self):
        self.detection = YOLO(detection_model_path)
        print("model loaded")

    def reset(self):
        self.color_map = defaultdict(lambda: defaultdict(int))
        self.max_heap_map = defaultdict(list)
        self.load_model()

    def add_color(self, ball_id, color):
        # Increment the color count
        self.color_map[ball_id][color] += 1

        # Clear the previous heap and rebuild it
        heap = []
        for color, count in self.color_map[ball_id].items():
            heapq.heappush(heap, (-count, color))  # Push negative count for max heap behavior

        self.max_heap_map[ball_id] = heap  # Update the heap for this ball_id

    def get_most_detected_color(self, ball_id):
        if ball_id not in self.max_heap_map:
            return None
        return self.max_heap_map[ball_id][0][1]  # Return the color with the highest count

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
        self.detection.track(frame, classes=[0], persist=True, conf=0.6)
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
            color = self.get_most_detected_color(id)
            if names[int(cls)] == 'ball':
                new_color = self.predict_color(img, x1, y1, x2, y2)
                self.add_color(id, new_color)
                most_detected = self.get_most_detected_color(id)
                if most_detected is not None:
                    color = most_detected

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


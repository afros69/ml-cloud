from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

from kolya_color import device, data_transforms_val, padding, best_model_wts
import cv2
import numpy as np
import PIL
import torch

app = FastAPI()


class MyStore:
    frames = []
    color_store = {}
    detection = None

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.detection = YOLO('yolov8-n-31.05.24.pt')

    def reset(self):
        self.frames = []
        self.color_store = {}
        self.load_model()


logic = MyStore()

color_map = {
    0: 'yellow',
    1: 'blue_str',
    2: 'red_str',
    3: 'pink_str',
    4: 'orange_str',
    5: 'green_str',
    6: 'brown_str',
    7: 'blue',
    8: 'red',
    9: 'pink',
    10: 'orange',
    11: 'green',
    12: 'brown',
    13: 'black',
    14: 'yellow_str',
    15: 'cue_ball',
}

number_map = {
    'yellow': 1,
    'blue_str': 10,
    'red_str': 11,
    'pink_str': 21,
    'orange_str': 13,
    'green_str': 14,
    'brown_str': 15,
    'blue': 2,
    'red': 3,
    'pink': 4,
    'orange': 5,
    'green': 6,
    'brown': 7,
    'black': 8,
    'yellow_str': 9,
    'cue_ball': 'cue_ball',
}


def predict_color(img, x1, y1, x2, y2):
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


def handle_frame(frame):
    results = logic.detection.track(frame, classes=[0], persist=True)
    if len(results) == 0:
        return None

    result_boxes = []

    result = results[0]
    boxes = result.boxes
    names = result.names
    ids = result.boxes.id
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    box_number = 1
    for box, alt_box, cls, conf, id in zip(boxes.data, boxes.xywh.data, boxes.cls, boxes.conf, ids.numpy()):
        x1, y1, x2, y2 = box[:4].int().tolist()
        x, y, w, h = alt_box[:4].int().tolist()
        stored = id in logic.color_store
        if names[int(cls)] == 'ball' and not stored:
            color = predict_color(img, x1, y1, x2, y2)
            if color.endswith("_str"):
                logic.color_store[id] = color
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


@app.post("/upload_frame")
async def upload_frame(frame: UploadFile = File(...)):
    frame_bytes = await frame.read()

    np_arr = np.frombuffer(frame_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    logic.frames.append(img)

    frame_size_mb = len(frame_bytes) / (1024 * 1024)

    print(f"Size of the frame: {frame_size_mb:.2f} MB")

    return handle_frame(img)


@app.get("/show_frames")
async def show_frames():
    for idx, frame in enumerate(logic.frames):
        boxes = handle_frame(frame)
        if not boxes or len(boxes) == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = box["color"]
            number = box["number"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{color} - {number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow(f"Frames", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    return {"message": f"Displayed {len(logic.frames)} frames"}


@app.post("/reset")
async def reset_frames():
    logic.reset()
    return True

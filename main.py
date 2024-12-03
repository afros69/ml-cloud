from time import sleep

from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from game import Game

games: dict[str, Game] = {}

app = FastAPI(
    title="Stripes ML Cloud",
)


@app.post("/upload_frame/{game_sid}")
async def upload_frame(game_sid: str, store_frames: bool = False, frame: UploadFile = File(...)):
    if game_sid not in games:
        games[game_sid] = Game(game_sid=game_sid, store_frames=store_frames)

    game = games[game_sid]

    frame_bytes = await frame.read()

    np_arr = np.frombuffer(frame_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if game.store_frames:
        game.frames.append(img)

    return game.handle_frame(img)


@app.get("/show_frames/{game_sid}")
async def show_frames(game_sid: str):
    game = games.get(game_sid, None)
    if not game:
        return
    frames = game.frames
    for idx, frame in enumerate(frames):
        boxes = game.handle_frame(frame)
        if not boxes or len(boxes) == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = box["color"]
            number = box["number"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{color} - {number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imwrite(f"{game.game_sid}.jpg", frame)
        cv2.imshow("Frames", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return {"message": f"Displayed {len(frames)} frames"}


@app.post("/reset/{game_sid}")
async def reset_frames(game_sid: str):
    game = games.get(game_sid, None)
    if not game:
        return False
    game.reset()
    return True

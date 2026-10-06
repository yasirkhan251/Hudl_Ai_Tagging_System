from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import easyocr
import torch
from ultralytics import YOLO

from capture import ScreenCapture, select_screen_roi
from court_filter import draw_court, foot_inside_court, select_court_roi
from config import CONFIDENCE, OCR_CONFIDENCE, OCR_EVERY_N_FRAMES, OUTPUT_DIR, YOLO_MODEL, WINDOW_NAME, MAX_PLAYERS
from player_memory import PlayerMemory


def clean_jersey(text: str):
    digits = re.sub(r"[^0-9]", "", text)
    if not digits or len(digits) > 2:
        return None
    value = int(digits)
    return str(value) if 0 <= value <= 99 else None


def read_jersey(reader, crop):
    if crop is None or crop.size == 0:
        return None, 0.0

    h, w = crop.shape[:2]
    if h < 20 or w < 10:
        return None, 0.0

    torso = crop[int(h * 0.15):int(h * 0.72)]
    if torso.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

    results = reader.readtext(
        gray,
        allowlist="0123456789",
        detail=1,
        paragraph=False,
        text_threshold=0.45,
        low_text=0.25,
        mag_ratio=1.5,
    )

    best = None
    for _, text, confidence in results:
        number = clean_jersey(text)
        confidence = float(confidence)
        if number is not None and confidence >= OCR_CONFIDENCE:
            if best is None or confidence > best[1]:
                best = (number, confidence)

    return best if best else (None, 0.0)


def draw_player(frame, box, track_id, memory):
    x1, y1, x2, y2 = box
    player = memory.get(track_id)
    jersey = player["jersey"] if player else None
    jersey_conf = player["jersey_confidence"] if player else 0.0

    label = f"ID {track_id}"
    if jersey is not None:
        label += f"  #{jersey}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 190, y1), (20, 30, 30), -1)
    cv2.putText(frame, label, (x1 + 5, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if jersey is not None:
        cv2.putText(frame, f"Jersey memory: {jersey_conf:.0%}",
                    (x1, min(frame.shape[0] - 8, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1, cv2.LINE_AA)


def main():
    print("HUDL AI TAGGING SYSTEM - PLAYER IDENTITY V1")
    print("Open Hudl first. Select only the video/player area.")
    print("Press Q in the AI window to stop.")

    model_path = Path(YOLO_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}. "
            "Copy yolov8n.pt into the models folder."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO...")
    model = YOLO(str(model_path))

    print("Loading EasyOCR...")
    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    print("Select the Hudl video region...")
    roi = select_screen_roi()
    print(f"Video ROI: {roi}")

    # Court coordinates are relative to the captured Hudl video frame.
    print("Select the full volleyball court area...")
    preview_capture = ScreenCapture()
    try:
        preview = preview_capture.grab(roi)
    finally:
        preview_capture.close()
    court = select_court_roi(preview)
    print(f"Court polygon points: {len(court)}")

    capture = ScreenCapture()
    memory = PlayerMemory()
    frame_index = 0
    last_fps_time = time.time()
    frames_since_fps = 0
    fps = 0.0
    memory_path = OUTPUT_DIR / "player_memory.json"

    try:
        while True:
            frame = capture.grab(roi)
            frame_index += 1
            frames_since_fps += 1

            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=CONFIDENCE,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                # Filter BEFORE OCR/memory: only people whose feet are inside
                # the calibrated court zone are treated as volleyball players.
                candidates = []
                for index, box in enumerate(xyxy):
                    if foot_inside_court(box, court):
                        candidates.append((index, box))

                # Safety limit. Keep the largest/highest-confidence court
                # detections if a bad frame contains more than 12 candidates.
                if len(candidates) > MAX_PLAYERS:
                    candidates.sort(key=lambda item: float(confs[item[0]]), reverse=True)
                    candidates = candidates[:MAX_PLAYERS]

                for index, box in candidates:
                    track_id = int(ids[index])
                    x1, y1, x2, y2 = map(int, box)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    crop = frame[y1:y2, x1:x2]
                    jersey, jersey_confidence = (None, 0.0)

                    if frame_index % OCR_EVERY_N_FRAMES == 0:
                        jersey, jersey_confidence = read_jersey(reader, crop)

                    memory.observe(
                        track_id,
                        jersey=jersey,
                        jersey_confidence=jersey_confidence,
                        bbox=(x1, y1, x2, y2),
                        detection_confidence=float(confs[index]),
                    )

                    draw_player(frame, (x1, y1, x2, y2), track_id, memory)

            memory.prune()

            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frames_since_fps / (now - last_fps_time)
                frames_since_fps = 0
                last_fps_time = now

            draw_court(frame, court)
            cv2.rectangle(frame, (0, 0), (390, 36), (15, 20, 25), -1)
            cv2.putText(frame, f"PLAYER AI | FPS {fps:.1f} | MAX {MAX_PLAYERS}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("c"), ord("C")):
                print("Recalibrating court polygon...")
                court = select_court_roi(frame)
                print(f"Court polygon points: {len(court)}")
            elif key in (ord("q"), ord("Q"), 27):
                break

    finally:
        capture.close()
        cv2.destroyAllWindows()

        with open(memory_path, "w", encoding="utf-8") as file:
            json.dump(memory.snapshot(), file, indent=2)

        print(f"Saved player memory: {memory_path}")


if __name__ == "__main__":
    main()

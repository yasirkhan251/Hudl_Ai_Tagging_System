from __future__ import annotations

import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import easyocr
import torch
from ultralytics import YOLO

from ball_detector import VolleyballDetector
from ball_tracker import BallTracker
from ball_capture import BallCaptureTool
from capture import ScreenCapture, select_screen_roi
from court_filter import draw_court, foot_inside_court, select_court_roi
from config import (
    CONFIDENCE,
    OCR_CONFIDENCE,
    OCR_EVERY_N_FRAMES,
    OUTPUT_DIR,
    YOLO_MODEL,
    WINDOW_NAME,
    MAX_PLAYERS,
    BALL_CONFIDENCE,
    BALL_MAX_MISSING,
    BALL_SMOOTHING,
    YOLO_IMGSZ,
    YOLO_HALF, YOLO_MAX_DET, BALL_MODEL, BALL_FRAME_DIR, BALL_CAPTURE_INTERVAL,
)
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
    gray = cv2.copyMakeBorder(
        gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
    )

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
    cv2.rectangle(
        frame, (x1, max(0, y1 - 28)), (x1 + 190, y1), (20, 30, 30), -1
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if jersey is not None:
        cv2.putText(
            frame,
            f"Jersey memory: {jersey_conf:.0%}",
            (x1, min(frame.shape[0] - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 120),
            1,
            cv2.LINE_AA,
        )


def main():
    print("HUDL AI TAGGING SYSTEM - V5 CUSTOM VOLLEYBALL DETECTOR")
    print("Open Hudl first. Select only the video/player area.")
    print("Press C to recalibrate court | B to draw/capture the volleyball with the mouse | Q to stop.")

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

    ball_detector = VolleyballDetector(
        BALL_MODEL,
        confidence=BALL_CONFIDENCE,
        imgsz=YOLO_IMGSZ,
        half=bool(YOLO_HALF and use_gpu),
    )
    BALL_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    print("Select the Hudl video region...")
    roi = select_screen_roi()
    print(f"Video ROI: {roi}")

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
    ball = BallTracker(max_missing=BALL_MAX_MISSING, smoothing=BALL_SMOOTHING)

    # Interactive ball capture/labeling. Press B, drag a tight box around
    # the volleyball, then press ENTER or S. This creates:
    #   images/train/*.jpg  -> full Hudl frame
    #   labels/train/*.txt  -> YOLO class-0 bounding box
    #   crops/*.jpg         -> tight volleyball crop for visual inspection
    ball_capture = None

    # OCR runs in a separate worker so a slow EasyOCR call does not stall
    # the capture/detection/display loop.
    ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
    ocr_jobs: dict[int, Future] = {}

    frame_index = 0
    last_fps_time = time.perf_counter()
    frames_since_fps = 0
    fps = 0.0

    inference_ms = 0.0
    capture_ms = 0.0
    ocr_pending = 0
    last_ball_capture = 0.0
    ball_frame_count = 0

    memory_path = OUTPUT_DIR / "player_memory.json"

    try:
        while True:
            capture_start = time.perf_counter()
            frame = capture.grab(roi)
            raw_frame = frame.copy()
            capture_ms = (time.perf_counter() - capture_start) * 1000.0

            frame_index += 1
            frames_since_fps += 1

            # Apply completed OCR results without blocking.
            completed_jobs = []
            for track_id, future in list(ocr_jobs.items()):
                if not future.done():
                    continue

                completed_jobs.append(track_id)
                try:
                    jersey, jersey_confidence = future.result()
                except Exception as exc:
                    print(f"OCR worker error for track {track_id}: {exc}")
                    jersey, jersey_confidence = None, 0.0

                player = memory.get(track_id)
                if player is not None and jersey is not None:
                    memory.observe(
                        track_id,
                        jersey=jersey,
                        jersey_confidence=jersey_confidence,
                        bbox=player["bbox"],
                        detection_confidence=player["detection_confidence"],
                    )

            for track_id in completed_jobs:
                ocr_jobs.pop(track_id, None)

            inference_start = time.perf_counter()
            # Player model handles ONLY people. Ball inference is separate.
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=CONFIDENCE,
                imgsz=YOLO_IMGSZ,
                half=bool(YOLO_HALF and use_gpu),
                max_det=YOLO_MAX_DET,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            result = results[0]
            boxes = result.boxes

            candidates = []

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                # Dedicated volleyball detector. It is independent of
                # player tracking and only runs when volleyball.pt exists.
                ball_candidates = ball_detector.detect(frame, court)

                if ball_candidates:
                    best_conf, best_box = max(
                        ball_candidates, key=lambda item: item[0]
                    )
                    ball.update(best_box, best_conf)
                else:
                    ball.mark_missing()

                # Players require tracker IDs.
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                if ids is not None:
                    for index, box in enumerate(xyxy):
                        if classes[index] != 0:
                            continue
                        if not foot_inside_court(box, court):
                            continue
                        candidates.append((index, box))

            if len(candidates) > MAX_PLAYERS:
                candidates.sort(
                    key=lambda item: float(confs[item[0]]), reverse=True
                )
                candidates = candidates[:MAX_PLAYERS]

            for index, box in candidates:
                track_id = int(ids[index])
                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                player = memory.observe(
                    track_id,
                    bbox=(x1, y1, x2, y2),
                    detection_confidence=float(confs[index]),
                )

                needs_ocr = (
                    player["jersey"] is None
                    or player["jersey_confidence"] < 0.80
                )

                # Only one OCR job per tracker at a time.
                if (
                    needs_ocr
                    and frame_index % OCR_EVERY_N_FRAMES == 0
                    and track_id not in ocr_jobs
                ):
                    crop = frame[y1:y2, x1:x2].copy()
                    ocr_jobs[track_id] = ocr_executor.submit(
                        read_jersey, reader, crop
                    )

                draw_player(frame, (x1, y1, x2, y2), track_id, memory)

            memory.prune()

            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                fps = frames_since_fps / (now - last_fps_time)
                frames_since_fps = 0
                last_fps_time = now

            ocr_pending = len(ocr_jobs)

            ball.draw(frame)
            draw_court(frame, court)

            cv2.rectangle(frame, (0, 0), (600, 86), (15, 20, 25), -1)

            if not ball_detector.available:
                ball_status = "MODEL NEEDED"
            else:
                ball_status = "TRACKED" if ball.visible else "LOST"
            cv2.putText(
                frame,
                f"AI | FPS {fps:.1f} | PLAYERS {len(candidates)} | BALL {ball_status}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"YOLO {inference_ms:.0f}ms | CAP {capture_ms:.1f}ms | OCR QUEUE {ocr_pending}",
                (10, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "C=Recalibrate | B=Ball Capture | Q=Exit",
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

            # Create the mouse tool only after the OpenCV window exists.
            if ball_capture is None:
                ball_capture = BallCaptureTool(WINDOW_NAME, BALL_DATASET_DIR)

            if ball_capture.active:
                ball_capture.draw(frame)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if ball_capture.active:
                if key in (13, 10, ord("s"), ord("S")):
                    saved = ball_capture.save()
                    if saved is not None:
                        last_ball_capture = time.perf_counter()
                elif key == 27:
                    ball_capture.cancel()

            elif key in (ord("b"), ord("B")):
                now_capture = time.perf_counter()
                if now_capture - last_ball_capture >= BALL_CAPTURE_INTERVAL:
                    ball_capture.start_capture(raw_frame)
                    print("Ball capture mode: drag a tight box around the volleyball, then press ENTER/S.")
                else:
                    remaining = BALL_CAPTURE_INTERVAL - (now_capture - last_ball_capture)
                    print(f"Ball capture cooldown: {remaining:.1f}s")

            if key in (ord("c"), ord("C")):
                print("Recalibrating court polygon...")
                court = select_court_roi(frame)
                print(f"Court polygon points: {len(court)}")
            elif key in (ord("q"), ord("Q"), 27):
                break

    finally:
        capture.close()
        cv2.destroyAllWindows()
        ocr_executor.shutdown(wait=False, cancel_futures=True)

        with open(memory_path, "w", encoding="utf-8") as file:
            json.dump(memory.snapshot(), file, indent=2)

        print(f"Saved player memory: {memory_path}")
        print("AI player tracking stopped.")


if __name__ == "__main__":
    main()

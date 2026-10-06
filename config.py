from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLO_MODEL = ROOT / "models" / "yolov8n.pt"
OUTPUT_DIR = ROOT / "output"

CONFIDENCE = 0.25
OCR_EVERY_N_FRAMES = 8
OCR_CONFIDENCE = 0.35
MAX_TRACK_AGE = 180
MAX_PLAYERS = 12

WINDOW_NAME = "Hudl AI Tagging System - Player Identity"

BALL_CONFIDENCE = 0.20
BALL_MAX_MISSING = 8
BALL_SMOOTHING = 0.65
YOLO_IMGSZ = 768
YOLO_HALF = True
YOLO_MAX_DET = 40

BALL_MODEL = ROOT / "models" / "volleyball.pt"
BALL_USE_GENERIC_FALLBACK = False
BALL_FRAME_DIR = ROOT / "data" / "ball_frames"
BALL_CAPTURE_INTERVAL = 0.5

# Interactive capture writes ready-to-train YOLO images/labels here.
BALL_DATASET_DIR = ROOT / "data" / "volleyball_dataset"

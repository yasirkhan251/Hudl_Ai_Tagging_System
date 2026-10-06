from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLO_MODEL = ROOT / "models" / "yolov8n.pt"
OUTPUT_DIR = ROOT / "output"

CONFIDENCE = 0.45
OCR_EVERY_N_FRAMES = 8
OCR_CONFIDENCE = 0.35
MAX_TRACK_AGE = 180
MAX_PLAYERS = 12

WINDOW_NAME = "Hudl AI Tagging System - Player Identity"

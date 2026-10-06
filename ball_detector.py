from pathlib import Path

import cv2
from ultralytics import YOLO


class VolleyballDetector:
    """
    Dedicated volleyball detector.

    If models/volleyball.pt exists, it is used as the primary detector.
    The model should be a one-class YOLO model where class 0 = volleyball.
    """

    def __init__(self, model_path: Path, confidence=0.20, imgsz=768, half=False):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.half = half
        self.model = None

        if self.model_path.exists():
            print(f"Loading custom volleyball model: {self.model_path}")
            self.model = YOLO(str(self.model_path))
        else:
            print("Custom volleyball model not found.")
            print("Ball detection is disabled until models/volleyball.pt is trained.")

    @property
    def available(self):
        return self.model is not None

    def detect(self, frame, court):
        if self.model is None:
            return []

        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Reuse the calibrated court polygon.
            inside = cv2.pointPolygonTest(
                court,
                (float(cx), float(cy)),
                False,
            )

            if inside >= 0:
                detections.append((float(confidence), box))

        return detections

from __future__ import annotations

import math
import time


class BallTracker:
    """Small stateful tracker for a volleyball detected by YOLO."""

    def __init__(self, max_missing=8, smoothing=0.65):
        self.max_missing = max_missing
        self.smoothing = smoothing
        self.x = None
        self.y = None
        self.radius = 0
        self.confidence = 0.0
        self.missing = 0
        self.last_seen = 0.0

    def update(self, box, confidence):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = max(2, int((x2 - x1 + y2 - y1) / 4))

        if self.x is None:
            self.x, self.y = cx, cy
        else:
            a = self.smoothing
            self.x = a * cx + (1.0 - a) * self.x
            self.y = a * cy + (1.0 - a) * self.y

        self.radius = radius
        self.confidence = float(confidence)
        self.missing = 0
        self.last_seen = time.time()

    def mark_missing(self):
        self.missing += 1

    @property
    def visible(self):
        return self.x is not None and self.missing <= self.max_missing

    def draw(self, frame):
        if not self.visible:
            return

        center = (int(self.x), int(self.y))
        cv2 = __import__("cv2")
        cv2.circle(frame, center, max(5, self.radius), (0, 165, 255), 2)
        cv2.circle(frame, center, 3, (0, 165, 255), -1)
        cv2.putText(
            frame,
            f"BALL {self.confidence:.0%}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

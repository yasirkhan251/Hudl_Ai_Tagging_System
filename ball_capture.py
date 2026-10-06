from __future__ import annotations

import time
from pathlib import Path

import cv2


class BallCaptureTool:
    """Interactive live-frame volleyball capture + YOLO annotation tool.

    B -> enter capture mode
    Mouse drag -> draw a tight box around the volleyball
    ENTER/S -> save full frame + YOLO label + ball crop
    ESC -> cancel
    """

    def __init__(self, window_name: str, output_dir: Path):
        self.window_name = window_name
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images" / "train"
        self.label_dir = self.output_dir / "labels" / "train"
        self.crop_dir = self.output_dir / "crops"

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir.mkdir(parents=True, exist_ok=True)

        self.active = False
        self.dragging = False
        self.start = None
        self.end = None
        self.frame = None
        self.saved_count = 0

        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def start_capture(self, frame):
        self.active = True
        self.dragging = False
        self.start = None
        self.end = None
        self.frame = frame.copy()

    def cancel(self):
        self.active = False
        self.dragging = False
        self.start = None
        self.end = None
        self.frame = None

    def _mouse_callback(self, event, x, y, flags, _param):
        if not self.active:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = (x, y)
            self.dragging = False

    def _normalised_box(self):
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])

        h, w = self.frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        return x1, y1, x2, y2

    def draw(self, frame):
        if not self.active:
            return

        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], 42), (20, 20, 20), -1
        )

        if self.start and self.end:
            x1, y1, x2, y2 = self._normalised_box()
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 255), 2
            )
            text = "BALL BOX | ENTER/S = SAVE | ESC = CANCEL"
        else:
            text = "BALL CAPTURE | Drag a tight box around the volleyball"

        cv2.putText(
            frame,
            text,
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def save(self) -> Path | None:
        if not self.active or self.frame is None:
            return None
        if not self.start or not self.end:
            return None

        x1, y1, x2, y2 = self._normalised_box()
        width = x2 - x1
        height = y2 - y1

        if width < 4 or height < 4:
            print("[BALL] Box is too small. Draw a larger box.")
            return None

        timestamp = int(time.time() * 1000)
        self.saved_count += 1
        stem = f"volleyball_{timestamp}_{self.saved_count:05d}"

        image_path = self.image_dir / f"{stem}.jpg"
        label_path = self.label_dir / f"{stem}.txt"
        crop_path = self.crop_dir / f"{stem}.jpg"

        cv2.imwrite(str(image_path), self.frame)

        crop = self.frame[y1:y2, x1:x2]
        cv2.imwrite(str(crop_path), crop)

        img_h, img_w = self.frame.shape[:2]
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        bw = width / img_w
        bh = height / img_h

        label_path.write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
            encoding="utf-8",
        )

        print(f"[BALL] Saved frame: {image_path}")
        print(f"[BALL] Saved label: {label_path}")
        print(f"[BALL] Saved crop:  {crop_path}")

        self.cancel()
        return image_path

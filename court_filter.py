import cv2
import numpy as np


def select_court_roi(frame):
    """
    Select the playable volleyball court area.

    The user draws a rectangle around the full court. Only detections whose
    bottom-center (foot point) lies inside this rectangle are accepted.
    """
    title = "Select VOLLEYBALL COURT - ENTER accept / ESC cancel"
    roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)

    x, y, w, h = map(int, roi)
    if w < 100 or h < 100:
        raise RuntimeError("Court ROI was cancelled or is too small.")

    return (x, y, x + w, y + h)


def foot_inside_court(box, court):
    """Return True when a person's foot point is inside the court ROI."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    foot_y = y2

    left, top, right, bottom = court
    return left <= cx <= right and top <= foot_y <= bottom


def draw_court(frame, court):
    left, top, right, bottom = court
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 180, 0), 2)
    cv2.putText(
        frame,
        "PLAYER COURT ZONE",
        (left + 5, max(22, top + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )

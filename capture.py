import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors

    def full_desktop(self):
        monitor = self.monitors[0]
        frame = np.array(self.sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame, (monitor["left"], monitor["top"])

    def grab(self, roi):
        frame = np.array(self.sct.grab(roi))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()


def select_screen_roi():
    cap = ScreenCapture()
    try:
        frame, origin = cap.full_desktop()
    finally:
        cap.close()

    title = "Select Hudl Video ROI - ENTER accept / ESC cancel"
    roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)
    if w < 100 or h < 100:
        raise RuntimeError("ROI was cancelled or is too small.")

    return {
        "left": origin[0] + x,
        "top": origin[1] + y,
        "width": w,
        "height": h,
    }

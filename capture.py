import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self, monitor_index=1):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors

        # MSS monitor 0 is the entire virtual desktop.
        # Physical monitors start at index 1.
        self.monitor_index = monitor_index

        if monitor_index < 1 or monitor_index >= len(self.monitors):
            raise RuntimeError(
                f"Monitor {monitor_index} not found. "
                f"Available physical monitors: 1-{len(self.monitors) - 1}"
            )

        self.monitor = self.monitors[monitor_index]

    def full_desktop(self):
        """Capture only the selected physical monitor."""
        monitor = self.monitor
        frame = np.array(self.sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame, (monitor["left"], monitor["top"])

    def grab(self, roi):
        """Capture only the selected ROI."""
        frame = np.array(self.sct.grab(roi))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()


def select_screen_roi():
    """
    Show only one physical monitor for ROI selection.

    MSS:
        monitor 0 = entire virtual desktop
        monitor 1 = first physical monitor
        monitor 2 = second physical monitor
        ...

    Change monitor_index below to 2 if your Hudl screen is on
    the second physical monitor.
    """
    monitor_index = 1
    cap = ScreenCapture(monitor_index=monitor_index)

    try:
        frame, origin = cap.full_desktop()
    finally:
        cap.close()

    title = "Select Hudl Video ROI - ENTER accept / ESC cancel"

    roi = cv2.selectROI(
        title,
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

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

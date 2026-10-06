import cv2
import numpy as np


def _fit_size(width, height, max_width=1400, max_height=780):
    """Return a display size that fits the current screen while preserving aspect ratio."""
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale)), scale


def select_court_roi(frame):
    """
    Select a volleyball court as a free-form polygon.

    The preview is explicitly fitted to the available screen area, so mouse
    coordinates always map back to the original captured frame correctly.

    Controls:
      Left click = add boundary point
      ENTER     = accept (minimum 4 points)
      R         = reset
      U         = undo last point
      ESC       = cancel

    Points close to the image edge are clamped to the image boundary. This
    makes it easier to select court corners that touch the edge of the video.
    """
    points = []
    original_h, original_w = frame.shape[:2]
    display_w, display_h, scale = _fit_size(original_w, original_h)

    # Use a fixed-size window so OpenCV/Windows resizing cannot introduce
    # ambiguous mouse-to-image coordinate mapping.
    window = "Select VOLLEYBALL COURT - Click boundary | ENTER accept | U undo | R reset | ESC cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_w, display_h)

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Map display coordinates back to original frame coordinates.
        ox = int(round(x / scale))
        oy = int(round(y / scale))

        # Hard clamp to the actual captured image.
        ox = max(0, min(original_w - 1, ox))
        oy = max(0, min(original_h - 1, oy))

        points.append((ox, oy))

    cv2.setMouseCallback(window, mouse_callback)

    while True:
        display = cv2.resize(
            frame,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA,
        )

        # Draw in display coordinates so the visual selection matches the
        # exact coordinate system used by the mouse callback.
        display_points = [
            (int(round(x * scale)), int(round(y * scale)))
            for x, y in points
        ]

        if display_points:
            for point in display_points:
                cv2.circle(display, point, 5, (0, 255, 255), -1)

            for i in range(1, len(display_points)):
                cv2.line(
                    display,
                    display_points[i - 1],
                    display_points[i],
                    (255, 180, 0),
                    2,
                )

            if len(display_points) >= 3:
                cv2.line(
                    display,
                    display_points[-1],
                    display_points[0],
                    (255, 180, 0),
                    2,
                )

                overlay = display.copy()
                cv2.fillPoly(
                    overlay,
                    [np.array(display_points, dtype=np.int32)],
                    (255, 180, 0),
                )
                display = cv2.addWeighted(overlay, 0.12, display, 0.88, 0)

        cv2.putText(
            display,
            f"COURT POINTS: {len(points)} | ENTER accept | U undo | R reset | ESC cancel",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            if len(points) >= 4:
                break
        elif key in (ord("u"), ord("U")):
            if points:
                points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Court polygon selection was cancelled.")

    cv2.destroyWindow(window)
    return np.array(points, dtype=np.int32)


def foot_inside_court(box, court):
    """Return True when a player's bottom-center foot point is inside the polygon."""
    x1, y1, x2, y2 = box
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)

    inside = cv2.pointPolygonTest(
        court,
        (float(foot_x), float(foot_y)),
        False,
    )
    return inside >= 0


def draw_court(frame, court):
    """Draw the calibrated court polygon."""
    if court is None or len(court) < 3:
        return

    cv2.polylines(
        frame,
        [court],
        isClosed=True,
        color=(255, 180, 0),
        thickness=2,
    )

    overlay = frame.copy()
    cv2.fillPoly(overlay, [court], (255, 180, 0))
    frame[:] = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

    x, y = court[0]
    cv2.putText(
        frame,
        "PLAYER COURT ZONE",
        (int(x) + 5, max(22, int(y) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )

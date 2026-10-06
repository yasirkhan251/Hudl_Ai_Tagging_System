import cv2
import numpy as np


def select_court_roi(frame):
    """
    Select a volleyball court as a free-form polygon.

    Left-click points around the playable court boundary.
    Press ENTER to accept, R to reset, ESC to cancel.

    A polygon is used instead of a rectangle because the court can appear
    trapezoidal or irregular due to camera perspective.
    """
    points = []
    window = "Select VOLLEYBALL COURT - Click boundary | ENTER accept | R reset | ESC cancel"

    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        display = frame.copy()

        if points:
            for point in points:
                cv2.circle(display, point, 5, (0, 255, 255), -1)

            for i in range(1, len(points)):
                cv2.line(display, points[i - 1], points[i], (255, 180, 0), 2)

            if len(points) >= 3:
                cv2.line(display, points[-1], points[0], (255, 180, 0), 2)
                overlay = display.copy()
                cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], (255, 180, 0))
                display = cv2.addWeighted(overlay, 0.12, display, 0.88, 0)

        cv2.putText(
            display,
            f"COURT POINTS: {len(points)} | ENTER=accept R=reset ESC=cancel",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if len(points) >= 4:
                break
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key == 27:  # Escape
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

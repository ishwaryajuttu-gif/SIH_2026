"""Draws the pipeline's real outputs on the video frame (boxes, hands, zones, links, status)."""
from __future__ import annotations

import cv2
import numpy as np

from .activity import ACTIVE, ActivityRecognizer, State
from .detector import Track
from .hands import FINGERTIPS, HAND_CONNECTIONS, Hand
from .interaction import HUMAN_LABELS, Interaction, Proximity

# BGR colours
C_OBJ = (230, 190, 60)
C_HUMAN = (180, 180, 180)
C_NEAR = (0, 200, 255)
C_CONTACT = (80, 220, 80)
C_MOVE = (255, 140, 60)
C_DANGER = (60, 60, 240)
C_WARN = (0, 190, 255)
C_WORK = (220, 200, 90)
C_HAND = (255, 255, 255)

STATE_COLOR = {
    State.IDLE: C_OBJ, State.HAND_NEAR: C_NEAR, State.INTERACTING: C_CONTACT,
    State.MANIPULATING: C_MOVE, State.COMPLETED: (200, 120, 255),
}


def _label(img, text, x, y, color, scale=0.45, fg=(20, 20, 20)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, th + 6)
    x = max(0, min(x, img.shape[1] - tw - 6))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def _dashed_poly(img, pts, color, thick=2, dash=10):
    n = len(pts)
    for i in range(n):
        p, q = np.array(pts[i], float), np.array(pts[(i + 1) % n], float)
        length = np.linalg.norm(q - p)
        steps = max(int(length // dash), 1)
        for k in range(0, steps, 2):
            a = p + (q - p) * (k / steps)
            b = p + (q - p) * (min(k + 1, steps) / steps)
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, thick, cv2.LINE_AA)


def draw(frame: np.ndarray, tracks: list[Track], hands: list[Hand], best: dict[int, Interaction],
         activity: ActivityRecognizer, zones: list[dict], restricted_objects: set[str],
         scene: dict | None = None) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # ---- zones ----
    shaded = [z for z in zones if z.get("enabled", True) and z.get("type") == "restricted" and len(z["points"]) >= 3]
    if shaded:
        layer = img.copy()
        for z in shaded:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in z["points"]], np.int32)
            cv2.fillPoly(layer, [pts], C_DANGER)
        alpha = 0.45 if any(z.get("active") for z in shaded) else 0.22
        img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    for z in zones:
        if len(z["points"]) < 3:
            continue
        pts = [(int(p[0] * w), int(p[1] * h)) for p in z["points"]]
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        if not z.get("enabled", True):
            _dashed_poly(img, pts, (120, 120, 120), 1)
            _label(img, f"{z['name']} (inactive)", x0, y0 + 18, (120, 120, 120), 0.42)
        elif z.get("type") == "workstation":
            _dashed_poly(img, pts, C_WORK, 2)
            _label(img, z["name"] + (" - hand inside" if z.get("active") else ""), x0, y0 + 18, C_WORK, 0.42)
        else:
            lvl = z.get("level", "NORMAL")
            color = C_DANGER if lvl != "NORMAL" else (80, 80, 220)
            cv2.polylines(img, [np.array(pts, np.int32)], True, color, 4 if z.get("active") else 2, cv2.LINE_AA)
            txt = z["name"]
            if lvl == "WARNING":
                txt = "HAND ENTERED RESTRICTED ZONE - WARNING"
            elif lvl == "CRITICAL":
                txt = "RESTRICTED ZONE - CRITICAL"
            _label(img, txt, x0, y0 + 18, color, 0.45, (255, 255, 255))

    # ---- objects ----
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        if t.label in HUMAN_LABELS:
            cv2.rectangle(img, (x1, y1), (x2, y2), C_HUMAN, 1, cv2.LINE_AA)
            _label(img, f"Person {t.conf:.2f}", x1, y1 + 16, C_HUMAN, 0.4)
            continue
        oa = activity.objects.get(t.track_id)
        state = oa.state if oa else State.IDLE
        color = STATE_COLOR.get(state, C_OBJ)
        if t.label in restricted_objects:
            color = C_DANGER if state in ACTIVE else (80, 80, 200)
        if not t.visible:  # temporarily not detected - box held by the tracker
            _dashed_poly(img, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], color, 2, 8)
            _label(img, f"#{t.track_id} {t.label} - OBJECT TEMPORARILY LOST", x1, y1, color, 0.42)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if state != State.IDLE else 2, cv2.LINE_AA)
        _label(img, f"#{t.track_id} {t.label} {t.conf:.2f} | {state.value}", x1, y1, color, 0.42)

    # ---- hand-object links ----
    for it in best.values():
        if it.level == Proximity.FAR:
            continue
        p = tuple(map(int, it.hand.landmarks[8]))
        c = tuple(map(int, it.track.center))
        color = C_CONTACT if it.level == Proximity.CONTACT else C_NEAR
        cv2.line(img, p, c, color, 2, cv2.LINE_AA)
        cv2.circle(img, c, 5, color, -1)

    # ---- hands ----
    for hd in hands:
        pts = [tuple(map(int, p)) for p in hd.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], C_HAND, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4 if i in FINGERTIPS else 2, C_CONTACT if i in FINGERTIPS else C_HAND, -1)
        cv2.circle(img, tuple(map(int, hd.palm_center)), 6, C_NEAR, 2)
        _label(img, f"{hd.handedness} {hd.score:.2f}", pts[0][0] - 20, pts[0][1] + 22, C_HAND, 0.4)

    # ---- status strip ----
    if scene:
        msgs = []
        if scene.get("hand_tracking_lost"):
            msgs.append(("HAND TRACKING LOST - holding state", C_WARN))
        if scene.get("safety_level") == "CRITICAL":
            msgs.append(("CRITICAL", C_DANGER))
        elif scene.get("safety_level") == "WARNING":
            msgs.append(("WARNING", C_WARN))
        y = 26   # top-right corner (the dashboard HUD uses the bottom corners)
        for text, color in msgs:
            _label(img, text, w - 10 - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] - 8, y, color,
                   0.55, (255, 255, 255) if color == C_DANGER else (20, 20, 20))
            y += 26
    return img

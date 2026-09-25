"""Small geometry helpers shared by the interaction, tracking and safety modules."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
Point = tuple[float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def center(b: Box) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def expand(b: Box, frac: float) -> Box:
    w, h = b[2] - b[0], b[3] - b[1]
    dx, dy = w * frac, h * frac
    return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)


def point_in_box(p: Point, b: Box) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def point_box_distance(p: Point, b: Box) -> float:
    """Euclidean distance from a point to a box (0 when inside)."""
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """Ray casting test. poly is a list of (x, y) in the same units as p."""
    x, y = p
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def any_point_in_polygon(points: Iterable[Point], poly: Sequence[Point]) -> bool:
    return any(point_in_polygon(p, poly) for p in points)

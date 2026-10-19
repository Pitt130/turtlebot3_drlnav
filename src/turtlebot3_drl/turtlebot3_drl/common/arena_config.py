"""Shared arena geometry for goal sampling and future USD scene construction."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def finite_number(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return number


@dataclass(frozen=True)
class BoxObstacle:
    x: float
    y: float
    size_x: float
    size_y: float
    yaw: float = 0.0

    def contains(self, x, y, margin=0.0):
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        return abs(local_x) <= self.size_x / 2 + margin and abs(local_y) <= self.size_y / 2 + margin

    def axis_aligned_corners(self, margin=0.0):
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        hx = c * (self.size_x / 2 + margin) + s * (self.size_y / 2 + margin)
        hy = s * (self.size_x / 2 + margin) + c * (self.size_y / 2 + margin)
        return [[self.x + hx, self.y + hy], [self.x + hx, self.y - hy],
                [self.x - hx, self.y - hy], [self.x - hx, self.y + hy]]


@dataclass(frozen=True)
class ArenaConfig:
    length: float
    width: float
    obstacles: tuple

    def goal_is_valid(self, x, y, margin=0.3):
        if not math.isfinite(x) or not math.isfinite(y):
            return False
        if abs(x) > self.length / 2 or abs(y) > self.width / 2:
            return False
        return not any(box.contains(x, y, margin) for box in self.obstacles)


def load_arena_config(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Arena config requires schema_version=1")
    boxes = []
    for i, item in enumerate(data["obstacles"]):
        if len(item["center"]) != 2 or len(item["size"]) != 2:
            raise ValueError(f"Obstacle {i}: center and size must have two values")
        boxes.append(BoxObstacle(
            finite_number(item["center"][0], "center x"),
            finite_number(item["center"][1], "center y"),
            finite_number(item["size"][0], "size x", positive=True),
            finite_number(item["size"][1], "size y", positive=True),
            finite_number(item.get("yaw", 0.0), "yaw"),
        ))
    return ArenaConfig(
        finite_number(data["arena_length"], "arena_length", positive=True),
        finite_number(data["arena_width"], "arena_width", positive=True), tuple(boxes))

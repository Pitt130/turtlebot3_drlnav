#!/usr/bin/env python3
"""Export planar box collisions from one SDF model into shared arena JSON."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def pose_of(element):
    pose = element.find("pose")
    if pose is None:
        return (0.0, 0.0, 0.0)
    if pose.get("relative_to") or pose.get("frame"):
        raise ValueError("Named SDF pose frames require explicit frame resolution")
    values = [float(v) for v in pose.text.split()]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid SDF pose")
    if abs(values[3]) > 1e-8 or abs(values[4]) > 1e-8:
        raise ValueError("Only planar, unpitched boxes are supported")
    return values[0], values[1], values[5]


def compose(parent, child):
    px, py, angle = parent
    cx, cy, yaw = child
    return (px + math.cos(angle) * cx - math.sin(angle) * cy,
            py + math.sin(angle) * cx + math.cos(angle) * cy, angle + yaw)


def export_arena(sdf_path, length, width):
    if not all(math.isfinite(v) and v > 0 for v in (length, width)):
        raise ValueError("Arena dimensions must be positive and finite")
    model = ET.parse(sdf_path).getroot().find("model")
    if model is None or model.findall("model"):
        raise ValueError("Expected one non-nested SDF model")
    obstacles = []
    for link in model.findall("link"):
        link_pose = compose(pose_of(model), pose_of(link))
        for collision in link.findall("collision"):
            size_text = collision.findtext("geometry/box/size")
            if size_text is None:
                raise ValueError(f"Unsupported collision geometry: {link.get('name')}")
            size = [float(v) for v in size_text.split()]
            if len(size) != 3 or not all(math.isfinite(v) and v > 0 for v in size):
                raise ValueError("Invalid collision box size")
            x, y, yaw = compose(link_pose, pose_of(collision))
            obstacles.append({"name": f"{link.get('name')}/{collision.get('name')}",
                              "center": [x, y], "size": size[:2], "yaw": yaw})
    return {"schema_version": 1, "arena_length": length, "arena_width": width,
            "obstacles": obstacles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path)
    parser.add_argument("--length", required=True, type=float)
    parser.add_argument("--width", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = export_arena(args.sdf, args.length, args.width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(data['obstacles'])} boxes to {args.out}")


if __name__ == "__main__":
    main()

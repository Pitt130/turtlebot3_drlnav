"""Simulation configuration without ROS, Gymnasium or simulator imports."""

import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def simulation_backend():
    backend = os.environ.get("DRLNAV_SIM_BACKEND", "gazebo").lower()
    if backend not in ("gazebo", "isaac"):
        raise ValueError("DRLNAV_SIM_BACKEND must be gazebo or isaac")
    return backend


def positive_int(value, name):
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def current_stage(stage_file="/tmp/drlnav_current_stage.txt"):
    """Return the stage of the running world.

    Gazebo launch files write the stage they loaded, so under Gazebo that file
    is authoritative and DRLNAV_STAGE may only repeat it; a mismatch would
    silently select another stage's goals. Isaac has no launch file yet, so
    there the explicit variable wins. An unknown stage raises instead of
    defaulting, which would mislabel sessions, weights and logs.
    """
    explicit = None
    if "DRLNAV_STAGE" in os.environ:
        explicit = positive_int(os.environ["DRLNAV_STAGE"], "DRLNAV_STAGE")
        if simulation_backend() == "isaac":
            return explicit
    try:
        launched = positive_int(Path(stage_file).read_text().strip(), "stage")
    except FileNotFoundError:
        if explicit is not None:
            return explicit
        raise FileNotFoundError(f"Simulation stage unknown: {stage_file} not found and "
                                "DRLNAV_STAGE not set; launch the Gazebo stage first") from None
    if explicit is not None and explicit != launched:
        raise ValueError(f"DRLNAV_STAGE={explicit} conflicts with stage {launched} "
                         f"written by the Gazebo launch file {stage_file}")
    return launched


def scan_count():
    if "DRLNAV_SCAN_SAMPLES" in os.environ:
        return positive_int(os.environ["DRLNAV_SCAN_SAMPLES"], "DRLNAV_SCAN_SAMPLES")
    if simulation_backend() == "isaac":
        return 40
    base_path = os.environ.get("DRLNAV_BASE_PATH")
    if not base_path:
        raise ValueError("Set DRLNAV_BASE_PATH for Gazebo or DRLNAV_SCAN_SAMPLES explicitly")
    path = Path(base_path) / "src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
    root = ET.parse(path).getroot()
    for link in root.find("model").findall("link"):
        if link.get("name") == "base_scan":
            return positive_int(link.findtext("sensor/ray/scan/horizontal/samples"), "scan samples")
    raise ValueError(f"No base_scan sensor in {path}")


def simulation_speed(stage):
    """Return the configured speed; Isaac's default is nominal, not measured RTF."""
    if "DRLNAV_SIM_SPEED" in os.environ:
        speed = float(os.environ["DRLNAV_SIM_SPEED"])
    elif simulation_backend() == "isaac":
        speed = 1.0
    else:
        base_path = os.environ.get("DRLNAV_BASE_PATH")
        if not base_path:
            raise ValueError("Set DRLNAV_BASE_PATH or DRLNAV_SIM_SPEED explicitly")
        path = Path(base_path) / f"src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_drl_stage{stage}/burger.model"
        speed = float(ET.parse(path).getroot().findtext("world/physics/real_time_factor"))
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Simulation speed must be positive and finite")
    return speed

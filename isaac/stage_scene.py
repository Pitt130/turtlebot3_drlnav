"""Build a TurtleBot3 DRL stage in Isaac Sim from the Gazebo world and model files.

Static models are made of planar SDF boxes. Moving obstacles are SDF cylinders
animated by obstacle_plugin/obstacleN.cc, a Gazebo PoseAnimation: keyframes
are offsets from the model's start pose, interpolated with ignition::math's
Catmull-Rom spline and looped. The parsing and interpolation below have no
Isaac dependency.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

GAZEBO_PKG = Path("src/turtlebot3_simulations/turtlebot3_gazebo")


def _pose(element):
    node = element.find("pose") if element is not None else None
    if node is None or not (node.text or "").strip():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if node.get("relative_to"):
        raise ValueError("Named pose frames are not supported")
    values = tuple(float(v) for v in node.text.split())
    if len(values) != 6:
        raise ValueError(f"Bad pose: {node.text!r}")
    return values


def _compose(parent, child):
    """2D frame composition plus z; roll and pitch must be zero."""
    for pose in (parent, child):
        if abs(pose[3]) > 1e-6 or abs(pose[4]) > 1e-6:
            raise ValueError("Only planar poses are supported")
    px, py, pz, _, _, pyaw = parent
    cx, cy, cz, _, _, cyaw = child
    c, s = math.cos(pyaw), math.sin(pyaw)
    return (px + c * cx - s * cy, py + s * cx + c * cy, pz + cz, 0.0, 0.0, pyaw + cyaw)


@dataclass
class Box:
    name: str
    center: tuple          # x, y, z
    size: tuple            # sx, sy, sz
    yaw: float


@dataclass
class KeyframeAnimation:
    """gazebo::common::PoseAnimation with ignition::math::Spline (tension 0)."""
    length: float
    loop: bool
    times: list
    points: list           # (x, y, z) offsets from the start pose
    yaws: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.points)
        closed = n > 1 and self.points[0] == self.points[-1]
        tangents = []
        for i in range(n):
            if i == 0:
                a, b = (self.points[n - 2], self.points[1]) if closed else (self.points[0], self.points[1])
            elif i == n - 1:
                if closed:
                    tangents.append(tangents[0])
                    continue
                a, b = self.points[i - 1], self.points[i]
            else:
                a, b = self.points[i - 1], self.points[i + 1]
            tangents.append(tuple(0.5 * (bb - aa) for aa, bb in zip(a, b)))
        self.tangents = tangents

    def offset(self, t):
        if self.loop and self.length > 0:
            t = math.fmod(t, self.length)
        t = min(max(t, self.times[0]), self.times[-1])
        i = max(0, min(len(self.times) - 2, next(k for k in range(len(self.times) - 1) if t <= self.times[k + 1])))
        span = self.times[i + 1] - self.times[i]
        u = (t - self.times[i]) / span if span > 0 else 0.0
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        p0, p1, m0, m1 = self.points[i], self.points[i + 1], self.tangents[i], self.tangents[i + 1]
        xyz = tuple(h00 * a + h10 * b + h01 * c + h11 * d for a, b, c, d in zip(p0, m0, p1, m1))
        yaw = self.yaws[i] + u * (self.yaws[i + 1] - self.yaws[i]) if self.yaws else 0.0
        return xyz, yaw


def parse_animation(cc_file):
    text = Path(cc_file).read_text()
    head = re.search(r'PoseAnimation\(\s*"[^"]*"\s*,\s*([-\d.]+)\s*,\s*(true|false)\s*\)', text)
    if not head:
        raise ValueError(f"No PoseAnimation in {cc_file}")
    num = r"([-\d.eE]+)"
    frames = re.findall(
        rf"CreateKeyFrame\(\s*{num}\s*\);\s*key->Translation\(ignition::math::Vector3d\(\s*{num}\s*,\s*{num}\s*,\s*{num}\s*\)\);"
        rf"\s*key->Rotation\(ignition::math::Quaterniond\(\s*{num}\s*,\s*{num}\s*,\s*([^)]+?)\s*\)\);", text)
    if len(frames) < 2:
        raise ValueError(f"Could not parse keyframes in {cc_file}")
    times = [float(f[0]) for f in frames]
    points = [(float(f[1]), float(f[2]), float(f[3])) for f in frames]
    yaws = [float(eval(f[6].replace("PI", repr(math.pi)), {"__builtins__": {}})) for f in frames]
    return KeyframeAnimation(float(head.group(1)), head.group(2) == "true", times, points, yaws)


@dataclass
class Obstacle:
    model: str
    body_name: str          # p3d body_name, the env reads "obstacleN" from child_frame_id
    start: tuple            # world pose of the model at t=0
    radius: float
    length: float
    animation: KeyframeAnimation | None

    @property
    def frame_id(self):
        # gazebo_ros_p3d reports the scoped link name, e.g. obstacle1::link_obstacle1.
        return f"{self.model}::{self.body_name}"

    def pose(self, t):
        if self.animation is None:
            return self.start[:3], self.start[5]
        (dx, dy, dz), dyaw = self.animation.offset(t)
        x, y, z, _, _, yaw = _compose(self.start, (dx, dy, dz, 0.0, 0.0, dyaw))
        return (x, y, z), yaw


@dataclass
class Stage:
    number: int
    boxes: list
    obstacles: list
    robot_start: tuple


def load_stage(repo, number):
    gz = Path(repo) / GAZEBO_PKG
    models = gz / "models"
    world = ET.parse(gz / "worlds" / f"turtlebot3_drl_stage{number}" / "burger.model").getroot().find("world")
    plugin_dir = models / "turtlebot3_drl_world" / "obstacle_plugin"
    boxes, obstacles, robot_start = [], [], (0.0,) * 6
    for include in world.iter("include"):
        uri = include.findtext("uri").strip()
        if not uri.startswith("model://"):
            continue
        name = uri[len("model://"):]
        if name in ("ground_plane", "sun"):
            continue
        place = _pose(include)
        if name == "turtlebot3_burger":
            robot_start = place
            continue
        model = ET.parse(models / name / "model.sdf").getroot().find("model")
        base = _compose(place, _pose(model))
        plugins = [p.get("filename") for p in include.findall("plugin")] + \
                  [p.get("filename") for p in model.findall("plugin")]
        p3d = model.find("plugin[@filename='libgazebo_ros_p3d.so']")
        if p3d is not None:
            link = model.find("link")
            cyl = link.find("collision/geometry/cylinder")
            anim_libs = [f for f in plugins if f and re.fullmatch(r"libobstacle\d+\.so", f)]
            animation = parse_animation(plugin_dir / (anim_libs[0][3:-3] + ".cc")) if anim_libs else None
            obstacles.append(Obstacle(name.split("/")[-1], p3d.findtext("body_name").strip(),
                                      _compose(base, _pose(link)), float(cyl.findtext("radius")),
                                      float(cyl.findtext("length")), animation))
            continue
        for link in model.findall("link"):
            link_pose = _compose(base, _pose(link))
            for col in link.findall("collision"):
                size = col.findtext("geometry/box/size")
                if size is None:
                    raise ValueError(f"{name}/{link.get('name')}: only box collisions are supported")
                x, y, z, _, _, yaw = _compose(link_pose, _pose(col))
                boxes.append(Box(f"{name.split('/')[-1]}_{link.get('name')}", (x, y, z),
                                 tuple(float(v) for v in size.split()), yaw))
    _attach_world_plugins(world, plugin_dir, obstacles)
    return Stage(number, boxes, obstacles, robot_start)


def _attach_world_plugins(world, plugin_dir, obstacles):
    """Some worlds list libobstacleN.so inside the obstacle's <include>; map any
    remaining ones in document order onto obstacles without an animation."""
    if all(o.animation is not None for o in obstacles):
        return
    libs = [p.get("filename") for p in world.iter("plugin")
            if p.get("filename") and re.fullmatch(r"libobstacle\d+\.so", p.get("filename"))]
    free = [o for o in obstacles if o.animation is None]
    for obstacle, lib in zip(free, libs[-len(free):] if libs else []):
        obstacle.animation = parse_animation(plugin_dir / (lib[3:-3] + ".cc"))


def build_in_isaac(world, stage, root="/World/stage"):
    """Create static boxes and kinematic obstacle cylinders; return a pose updater."""
    import numpy as np
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import SingleXFormPrim
    from pxr import Gf, UsdGeom, UsdPhysics

    for i, b in enumerate(stage.boxes):
        q = (math.cos(b.yaw / 2), 0.0, 0.0, math.sin(b.yaw / 2))
        world.scene.add(FixedCuboid(prim_path=f"{root}/walls/{b.name}", name=f"wall_{i}_{b.name}",
                                    position=np.array(b.center), orientation=np.array(q),
                                    scale=np.array(b.size), color=np.array([0.55, 0.4, 0.25])))
    movers = []
    for o in stage.obstacles:
        path = f"{root}/obstacles/{o.model}"
        UsdGeom.Xform.Define(world.stage, path)
        cyl = UsdGeom.Cylinder.Define(world.stage, f"{path}/body")
        cyl.CreateRadiusAttr(o.radius)
        cyl.CreateHeightAttr(o.length)
        cyl.CreateAxisAttr("Z")
        cyl.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.25)])
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        body = UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(path))
        body.CreateKinematicEnabledAttr(True)
        prim = SingleXFormPrim(path, name=f"obstacle_{o.model}")
        movers.append((o, prim))

    def update(t):
        for o, prim in movers:
            (x, y, z), yaw = o.pose(t)
            prim.set_world_pose(position=np.array([x, y, z]),
                                orientation=np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
    update(0.0)
    return update

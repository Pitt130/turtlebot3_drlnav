"""Import the TurtleBot3 Burger URDF into USD and check its differential drive.

Run with Isaac Sim's Python, after sourcing ROS 2 Jazzy:
    ~/isaacsim/.venv/bin/python isaac/build_turtlebot3.py \
        --description ~/isaac_assets/turtlebot3/turtlebot3_description \
        --out ~/isaac_assets/turtlebot3_burger_usd

The drive check commands wheel velocities from (v, w) exactly as a
differential controller does and compares the base motion with the command.
"""
import argparse
import math
from pathlib import Path
import re

WHEEL_RADIUS = 0.033      # m, turtlebot3_burger.urdf and Gazebo model.sdf
WHEEL_SEPARATION = 0.160  # m
WHEEL_JOINTS = ("wheel_left_joint", "wheel_right_joint")


def strip_xacro(source, target):
    """The ROBOTIS URDF only uses xacro for an optional joint/link namespace."""
    text = Path(source).read_text()
    text = re.sub(r"\s*<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\s*<xacro:(arg|property)\b[^>]*/>", "", text)
    text = text.replace("${namespace}", "").replace(' xmlns:xacro="http://ros.org/wiki/xacro"', "")
    if "${" in text or "xacro:" in text:
        raise ValueError("Unexpected xacro construct left in the URDF")
    Path(target).write_text(text)


def wheel_speeds(v, w):
    half = WHEEL_SEPARATION / 2
    return (v - w * half) / WHEEL_RADIUS, (v + w * half) / WHEEL_RADIUS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--damping", type=float, default=1.0e4)
    parser.add_argument("--physics-hz", type=float, default=200.0)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    app.update()
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    # Inside urdf/ so the importer resolves package://turtlebot3_description/meshes.
    urdf = args.description / "urdf" / "turtlebot3_burger_isaac.urdf"
    strip_xacro(args.description / "urdf" / "turtlebot3_burger.urdf", urdf)
    args.out.mkdir(parents=True, exist_ok=True)
    config = URDFImporterConfig(
        urdf_path=str(urdf), usd_path=str(args.out), fix_base=False,
        joint_drive_type="force", joint_target_type="velocity",
        override_joint_stiffness=0.0, override_joint_damping=args.damping,
    )
    usd_file = URDFImporter(config).import_urdf()
    print(f"USD written: {usd_file}", flush=True)

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

    world = World(physics_dt=1.0 / args.physics_hz, rendering_dt=1.0 / args.physics_hz)
    world.scene.add_default_ground_plane()
    prim = add_reference_to_stage(usd_path=usd_file, prim_path="/World/burger")
    # The importer puts physics in an unselected "Physics" variant; the PhysX
    # variant layers PhysX settings over the generic physics payload.
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    from pxr import UsdPhysics
    stage = world.stage
    root = next(p.GetPath().pathString for p in stage.Traverse()
                if p.GetPath().pathString.startswith("/World/burger") and p.HasAPI(UsdPhysics.ArticulationRootAPI))
    robot = world.scene.add(SingleArticulation(prim_path=root, name="burger"))
    world.reset()
    print(f"articulation root: {root}; dof names: {robot.dof_names}", flush=True)
    wheels = [robot.get_dof_index(name) for name in WHEEL_JOINTS]

    def yaw(q):
        w, x, y, z = q
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def run(v, w, seconds):
        for _ in range(int(0.5 * args.physics_hz)):   # settle
            world.step(render=False)
        p0, q0 = robot.get_world_pose()
        left, right = wheel_speeds(v, w)
        targets = [0.0] * robot.num_dof
        targets[wheels[0]], targets[wheels[1]] = left, right
        for _ in range(int(seconds * args.physics_hz)):
            robot.apply_action(ArticulationAction(joint_velocities=targets))
            world.step(render=False)
        p1, q1 = robot.get_world_pose()
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        turn = math.remainder(yaw(q1) - yaw(q0), 2 * math.pi)
        robot.apply_action(ArticulationAction(joint_velocities=[0.0] * robot.num_dof))
        return dist, turn, float(p1[2])

    for v, w, seconds in [(0.2, 0.0, 3.0), (0.0, 1.0, 3.0), (0.1, 0.5, 4.0)]:
        dist, turn, z = run(v, w, seconds)
        expect_d = abs(v) * seconds if w == 0 else abs(2 * v / w * math.sin(w * seconds / 2))
        print(f"cmd v={v:.2f} m/s w={w:.2f} rad/s for {seconds:.0f}s: moved {dist:.3f} m "
              f"(expect {expect_d:.3f}), turned {math.degrees(turn):.1f} deg "
              f"(expect {math.degrees(math.remainder(w * seconds, 2 * math.pi)):.1f}), base z={z:.3f}",
              flush=True)

    app.close()


if __name__ == "__main__":
    main()

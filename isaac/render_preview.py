"""Render preview images of the TurtleBot3 in Isaac Sim (headless).

    ~/isaacsim/.venv/bin/python isaac/render_preview.py --usd <burger.usda> --out <dir>
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 1280, "height": 720})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from build_turtlebot3 import use_sphere_wheel_collisions

    world = World(physics_dt=1 / 200, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()
    prim = add_reference_to_stage(usd_path=args.usd, prim_path="/World/burger")
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    use_sphere_wheel_collisions(world.stage, "/World/burger")
    for name, pos, size in [("wall_x", (0.6, 0.0, 0.25), (0.15, 1.0, 0.5)),
                            ("wall_y", (0.0, 0.7, 0.25), (1.0, 0.15, 0.5))]:
        world.scene.add(FixedCuboid(prim_path=f"/World/{name}", name=name, position=np.array(pos),
                                    scale=np.array(size), color=np.array([0.55, 0.4, 0.25])))
    world.reset()
    for _ in range(100):
        world.step(render=True)

    args.out.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    for name, eye, target in [("perspective", (-0.55, -0.55, 0.45), (0.05, 0.05, 0.08)),
                              ("top", (0.15, 0.2, 1.6), (0.15, 0.2, 0.0))]:
        set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path="/OmniverseKit_Persp")
        for _ in range(60):
            world.step(render=True)
        path = args.out / f"tb3_{name}.png"
        capture_viewport_to_file(viewport, str(path))
        for _ in range(20):
            app.update()
        print("saved", path, flush=True)
    app.close()


if __name__ == "__main__":
    main()

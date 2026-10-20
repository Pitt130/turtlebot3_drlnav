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
    parser.add_argument("--stage", type=int, help="render turtlebot3_drl_stage<N> instead of the test walls")
    parser.add_argument("--time", type=float, default=0.0, help="obstacle animation time to show")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
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
    from stage_scene import build_in_isaac, load_stage

    world = World(physics_dt=1 / 200, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()
    prim = add_reference_to_stage(usd_path=args.usd, prim_path="/World/burger")
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    use_sphere_wheel_collisions(world.stage, "/World/burger")
    move = build_in_isaac(world, load_stage(args.repo, args.stage)) if args.stage else None
    for name, pos, size in ([] if args.stage else [("wall_x", (0.6, 0.0, 0.25), (0.15, 1.0, 0.5)),
                                                    ("wall_y", (0.0, 0.7, 0.25), (1.0, 0.15, 0.5))]):
        world.scene.add(FixedCuboid(prim_path=f"/World/{name}", name=name, position=np.array(pos),
                                    scale=np.array(size), color=np.array([0.55, 0.4, 0.25])))
    world.reset()
    for _ in range(100):
        if move:
            move(args.time)
        world.step(render=True)

    args.out.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    views = [("perspective", (-0.55, -0.55, 0.45), (0.05, 0.05, 0.08)), ("top", (0.15, 0.2, 1.6), (0.15, 0.2, 0.0))]
    if args.stage:
        views = [("stage_top", (0.0, -0.01, 7.0), (0.0, 0.0, 0.0)), ("stage_perspective", (-3.6, -3.6, 2.6), (0.0, 0.0, 0.0))]
    for name, eye, target in views:
        set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path="/OmniverseKit_Persp")
        for _ in range(60):
            world.step(render=True)
        path = args.out / (f"stage{args.stage}_{name}.png" if args.stage else f"tb3_{name}.png")
        capture_viewport_to_file(viewport, str(path))
        for _ in range(20):
            app.update()
        print("saved", path, flush=True)
    app.close()


if __name__ == "__main__":
    main()

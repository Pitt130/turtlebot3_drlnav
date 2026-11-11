"""Load a GRScenes apartment through grscene.load_grscene and render a top view and a room view.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/.venv/bin/python isaac/render_grscene.py --scene <dir> --out <dir>
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # grscene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.scene.resolve()
    # Only the shared Materials library goes on the MDL search path. The scene folder links to every
    # downloaded model and each model folder links back to Materials, so Kit's scan of those trees
    # grows with every scene fetched (31 GB and an OOM kill with six scenes on disk).
    mdl_dir = root.parents[2] / "mdl_search"
    mdl_dir.mkdir(exist_ok=True)
    if not (mdl_dir / "Materials").exists():
        (mdl_dir / "Materials").symlink_to(root.parents[1] / "Materials")
    mdl_paths = [str(mdl_dir)]
    os.environ["MDL_SYSTEM_PATH"] = os.pathsep.join(mdl_paths)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 1600, "height": 1000, "renderer": "RaytracedLighting", "anti_aliasing": 2,
                         "extra_args": ["--/renderer/mdl/searchPaths/templates=" + ";".join(mdl_paths)]})
    from isaacsim.core.api import World
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from grscene import load_grscene

    world = World(physics_dt=1 / 200, rendering_dt=1 / 60)
    scene = load_grscene(world, args.scene)
    world.reset()
    cx, cy = scene.center
    for name, eye, target in (("top", (cx, cy - 0.01, 1.6 * scene.half_width + 3), (cx, cy, 0.0)),
                              ("room", (cx - 1.0, cy - 3.0, 1.4), (cx + 1.5, cy + 1.5, 0.3))):
        set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path="/OmniverseKit_Persp")
        for _ in range(120):
            world.render()
        capture_viewport_to_file(get_active_viewport(), str(args.out / f"{name}.png"))
        for _ in range(20):
            app.update()
        print("saved", args.out / f"{name}.png", flush=True)
    app.close()


if __name__ == "__main__":
    main()

"""Real indoor scenes (GRScenes) for the Isaac runtime.

GRScenes (InternRobotics, CC BY-NC-SA 4.0) ships OpenUSD apartments authored in
centimetres, with triangle-mesh colliders on walls, floors and furniture in
start_result_navigation.usd. load_grscene() references one into the metre-based
stage, build_occupancy() scans it at the lidar height with physics raycasts, and
StartGoalSampler draws start and goal poses from the free cells. Goals still go
out on goal_pose, so the environment and training code are unchanged.

The occupancy math has no Isaac dependency (used by tests and tools).
"""
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

LIDAR_HEIGHT = 0.172        # base_scan above the floor
FREE, OCCUPIED, OUTSIDE, LOW = 0, 1, 2, 3   # LOW: an object below the lidar plane (shoes, steps)


def _dilate(mask, n, pad):
    """Boolean disc dilation by n cells; `pad` is the value assumed outside the grid."""
    padded = np.pad(mask, n, constant_values=pad)
    out = np.zeros_like(mask)
    rows, cols = mask.shape
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            if dr * dr + dc * dc <= n * n:
                out |= padded[n + dr:n + dr + rows, n + dc:n + dc + cols]
    return out


@dataclass
class Occupancy:
    """Cells at the lidar height: FREE, OCCUPIED (something within one cell at the lidar height),
    LOW (something on the floor but entirely below the lidar plane) or OUTSIDE (no floor)."""
    origin: tuple           # world (x, y) of the centre of cell [0, 0]
    cell: float             # metres
    grid: np.ndarray        # [row (y), col (x)] uint8

    @property
    def shape(self):
        return self.grid.shape

    def to_cell(self, x, y):
        return (int(round((y - self.origin[1]) / self.cell)), int(round((x - self.origin[0]) / self.cell)))

    def to_world(self, row, col):
        return (self.origin[0] + col * self.cell, self.origin[1] + row * self.cell)

    def inflated(self, radius):
        """Mark every cell within `radius` of an obstacle or the outside as OCCUPIED (robot footprint)."""
        n = int(math.ceil(radius / self.cell))
        blocked = self.grid != FREE
        padded = np.pad(blocked, n, constant_values=True)
        grown = np.zeros_like(blocked)
        rows, cols = blocked.shape
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                if dr * dr + dc * dc <= n * n:
                    grown |= padded[n + dr:n + dr + rows, n + dc:n + dc + cols]
        out = self.grid.copy()
        out[grown & (self.grid == FREE)] = OCCUPIED
        return Occupancy(self.origin, self.cell, out)

    def inflated_split(self, radius, low_radius):
        """Like inflated(), but LOW cells get their own margin: the lidar cannot see them, so the
        planner has to keep the robot farther away than from walls the policy avoids by itself."""
        visible = (self.grid != FREE) & (self.grid != LOW)
        grown = _dilate(visible, int(math.ceil(radius / self.cell)), True) | \
            _dilate(self.grid == LOW, int(math.ceil(low_radius / self.cell)), False)
        out = self.grid.copy()
        out[grown & (self.grid == FREE)] = OCCUPIED
        return Occupancy(self.origin, self.cell, out)

    def inflated_open(self, radius, low_radius, open_radius=None, open_low_radius=None, min_area=0.5):
        """inflated_split(radius, low_radius) with narrow passages opened. A free region of at least
        `min_area` m2 that the margins cut off from the largest one, but that reaches it through cells at
        least `open_radius` from obstacles (`open_low_radius` from LOW cells), is joined to it along its
        shortest such 4-connected corridor, widened by 2 cells where the smaller margins allow. Returns
        (grid, passage) with the opened cells FREE in the grid and True in `passage`."""
        grid = self.inflated_split(radius, low_radius)
        passage = np.zeros(self.grid.shape, dtype=bool)
        if not open_radius:
            return grid, passage
        small = self.inflated_split(open_radius, open_low_radius or open_radius).grid == FREE
        labels, sizes = grid.components()
        if not sizes:
            return grid, passage
        main = int(np.argmax(sizes))
        joined = labels == main
        others = {k for k, n in enumerate(sizes) if k != main and n * self.cell ** 2 >= min_area}
        rows, cols = self.grid.shape
        while others:
            seen = joined.copy()                       # breadth-first from everything joined so far
            parent = {}
            queue = deque(map(tuple, np.argwhere(joined)))
            hit = None
            while queue and hit is None:
                r, c = queue.popleft()
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= rr < rows and 0 <= cc < cols and small[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        parent[(rr, cc)] = (r, c)
                        if labels[rr, cc] in others:
                            hit = (rr, cc)
                            break
                        queue.append((rr, cc))
            if hit is None:
                break
            path = np.zeros(self.grid.shape, dtype=bool)
            cell = parent[hit]
            while not joined[cell]:
                path[cell] = True
                cell = parent[cell]
            corridor = (_dilate(path, 2, False) & small & (grid.grid != FREE)) | path
            passage |= corridor
            joined |= corridor | (labels == labels[hit])
            others.discard(int(labels[hit]))
        grid.grid[passage] = FREE
        return grid, passage

    def components(self):
        """Label 4-connected FREE regions; -1 elsewhere. Returns (labels, sizes)."""
        labels = np.full(self.grid.shape, -1, dtype=np.int32)
        sizes = []
        rows, cols = self.grid.shape
        for r0 in range(rows):
            for c0 in range(cols):
                if self.grid[r0, c0] != FREE or labels[r0, c0] >= 0:
                    continue
                label = len(sizes)
                queue, count = deque([(r0, c0)]), 0
                labels[r0, c0] = label
                while queue:
                    r, c = queue.popleft()
                    count += 1
                    for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if 0 <= rr < rows and 0 <= cc < cols and self.grid[rr, cc] == FREE and labels[rr, cc] < 0:
                            labels[rr, cc] = label
                            queue.append((rr, cc))
                sizes.append(count)
        return labels, sizes

    def path_steps(self, row, col, max_steps):
        """8-connected wavefront distance (in cells) from a cell through FREE cells, inf beyond max_steps."""
        free = self.grid == FREE
        dist = np.full(self.grid.shape, np.inf)
        if not free[row, col]:
            return dist
        frontier = np.zeros_like(free)
        frontier[row, col] = True
        dist[row, col] = 0
        for k in range(1, max_steps + 1):
            p = np.pad(frontier, 1)
            grown = np.zeros_like(frontier)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        grown |= p[1 + dr:1 + dr + free.shape[0], 1 + dc:1 + dc + free.shape[1]]
            new = grown & free & np.isinf(dist)
            if not new.any():
                break
            dist[new] = k
            frontier = new
        return dist

    def save(self, path):
        np.savez_compressed(path, origin=np.array(self.origin), cell=self.cell, grid=self.grid)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(tuple(float(v) for v in d["origin"]), float(d["cell"]), d["grid"])

    def to_image(self, marks=()):
        """RGB array for a quick look; marks are (x, y, (r, g, b)) in world coordinates."""
        img = np.zeros((*self.grid.shape, 3), dtype=np.uint8)
        img[self.grid == FREE] = (235, 235, 235)
        img[self.grid == OCCUPIED] = (30, 30, 30)
        img[self.grid == OUTSIDE] = (90, 120, 160)
        img[self.grid == LOW] = (230, 140, 30)
        for x, y, color in marks:
            r, c = self.to_cell(x, y)
            img[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = color
        return img[::-1]      # image up = +y


class StartGoalSampler:
    """Start and goal poses from the free space of an occupancy grid.

    Cells are taken from the largest connected free region of the grid inflated
    by `inflate` (robot radius plus margin). A goal lies `goal_min`..`goal_max` m
    (Euclidean) from where the robot is, at most `path_max` m away along the
    8-connected grid path (so the 50 s episode can reach it), and at least
    `separation` m from the previous goal, as in the stage goal lists.
    """

    def __init__(self, occupancy, inflate=0.25, goal_min=1.5, goal_max=5.5, path_max=7.0, separation=2.0, seed=None,
                 low_inflate=None, open_passages=None):
        """`low_inflate` gives LOW cells their own margin and `open_passages` (radius, low radius) joins regions
        the margins cut off through narrower passages (Occupancy.inflated_open), as the waypoint planner does,
        so every start and goal is reachable on the planner's grid. Starts and goals are never placed in an
        opened passage: they keep the full margin."""
        passage = None
        if low_inflate is None and not open_passages:
            self.occ = occupancy.inflated(inflate)
        else:
            self.occ, passage = occupancy.inflated_open(inflate, inflate if low_inflate is None else low_inflate,
                                                        *(open_passages or (None, None)))
        self.goal_min, self.goal_max, self.path_max, self.separation = goal_min, goal_max, path_max, separation
        labels, sizes = self.occ.components()
        if not sizes:
            raise RuntimeError("no free space in the occupancy grid")
        self.main = int(np.argmax(sizes))
        self.occ.grid[(self.occ.grid == FREE) & (labels != self.main)] = OCCUPIED   # closets, balconies etc.
        free = self.occ.grid == FREE
        self.cells = np.argwhere(free if passage is None else free & ~passage)
        self.rng = np.random.default_rng(seed)
        self.area = len(self.cells) * self.occ.cell ** 2

    def sample_start(self):
        """(x, y, yaw) uniformly over the free cells."""
        row, col = self.cells[self.rng.integers(len(self.cells))]
        x, y = self.occ.to_world(row, col)
        return x, y, float(self.rng.uniform(-math.pi, math.pi))

    def sample_goal(self, from_xy, previous_goal=None):
        row, col = self.occ.to_cell(*from_xy)
        row, col = min(max(row, 0), self.occ.shape[0] - 1), min(max(col, 0), self.occ.shape[1] - 1)
        if self.occ.grid[row, col] != FREE:   # robot stopped next to furniture: start the wavefront at the nearest free cell
            d = np.hypot(self.cells[:, 0] - row, self.cells[:, 1] - col)
            row, col = self.cells[int(np.argmin(d))]
        steps = self.occ.path_steps(row, col, int(self.path_max / self.occ.cell))
        xy = np.array([self.occ.to_world(r, c) for r, c in self.cells])
        euclid = np.hypot(xy[:, 0] - from_xy[0], xy[:, 1] - from_xy[1])
        reach = steps[self.cells[:, 0], self.cells[:, 1]] * self.occ.cell
        ok = (euclid >= self.goal_min) & (euclid <= self.goal_max) & (reach <= self.path_max)
        if previous_goal is not None:
            far = np.hypot(xy[:, 0] - previous_goal[0], xy[:, 1] - previous_goal[1]) >= self.separation
            if (ok & far).any():
                ok &= far
        if not ok.any():                        # dead end: drop the distance floor, then everything but reachability
            ok = (euclid <= self.goal_max) & (reach <= self.path_max) & (euclid >= 0.5)
        if not ok.any():
            ok = np.isfinite(reach) & (euclid >= 0.5)
        choice = self.rng.choice(np.flatnonzero(ok))
        return float(xy[choice, 0]), float(xy[choice, 1])


@dataclass
class GRScene:
    prim_path: str
    lo: np.ndarray          # collider bounds, metres
    hi: np.ndarray
    scene_dir: Path

    @property
    def center(self):
        return ((self.lo[0] + self.hi[0]) / 2, (self.lo[1] + self.hi[1]) / 2)

    @property
    def half_width(self):
        return max(self.hi[0] - self.lo[0], self.hi[1] - self.lo[1]) / 2 + 0.3


def load_grscene(world, scene_dir, usd_name="start_result_navigation.usd", prim_path="/World/grscene",
                 hide_ceilings=True, dome_intensity=1000.0, drive_over=("blanket",), drive_over_height=0.05):
    """Reference the scene (centimetres -> metres), disable its own physics scene, return its collider bounds.

    Ceilings are hidden (render only; their colliders are 2.8 m up and irrelevant to the robot) so
    top-down views show the rooms, and a dome light is added because the scene's own lights are dim.
    Colliders of `drive_over` categories that lie within `drive_over_height` of the floor (rugs, which
    the dataset files under "blanket") are disabled: a 1.5 cm rug edge stops the Burger's 1 cm caster
    ball in PhysX, where a real rug gives way, and it tilts the robot into the tumble test.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    scene_dir = Path(scene_dir)
    # GRScenes layers do not author metersPerUnit; USD's fallback for that is centimetres.
    layer_mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(str(scene_dir / usd_name), Usd.Stage.LoadNone))
    scale = layer_mpu / UsdGeom.GetStageMetersPerUnit(world.stage)
    # Scale on a parent xform, so the layer's own transforms (in its units) stay inside it.
    parent = UsdGeom.Xform.Define(world.stage, prim_path)
    parent.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    prim = add_reference_to_stage(usd_path=str(scene_dir / usd_name), prim_path=f"{prim_path}/scene")
    rewritten, ceilings = 0, 0
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.Scene):             # the world already has one; two would split the colliders
            p.SetActive(False)
        if hide_ceilings and "/ceiling/" in str(p.GetPath()) and p.IsA(UsdGeom.Imageable) \
                and "/ceiling/" not in str(p.GetParent().GetPath()):
            UsdGeom.Imageable(p).MakeInvisible()
            ceilings += 1
        # Shaders point at "./Materials/<name>.mdl"; Kit 110 cannot resolve that relative module
        # form ("could not find module ::.::Materials::<name>"), so make the paths absolute.
        attr = p.GetAttribute("info:mdl:sourceAsset")
        if attr and attr.IsValid():
            asset = attr.Get()
            path = asset.path if asset is not None else ""
            if path.startswith("./") or path.startswith("../") or (path and not path.startswith("/") and "://" not in path):
                target = (scene_dir / path).resolve()
                if target.exists():
                    attr.Set(Sdf.AssetPath(str(target)))
                    rewritten += 1
    if rewritten or ceilings:
        print(f"grscene: {rewritten} MDL asset paths made absolute, {ceilings} ceiling prims hidden", flush=True)
    if dome_intensity > 0:
        dome = UsdLux.DomeLight.Define(world.stage, f"{prim_path}_dome")
        dome.CreateIntensityAttr(dome_intensity)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    los, his, backdrops = [], [], 0
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            box = cache.ComputeWorldBound(p).ComputeAlignedBox()
            if not box.IsEmpty():
                los.append(np.array(box.GetMin())); his.append(np.array(box.GetMax()))
        elif p.IsA(UsdGeom.Mesh):
            # The scenes carry a 500 m aerial-photo backdrop far below the floor; hide it.
            box = cache.ComputeWorldBound(p).ComputeAlignedBox()
            if not box.IsEmpty() and max(np.array(box.GetMax()) - np.array(box.GetMin())) > 60:
                UsdGeom.Imageable(p).MakeInvisible()
                backdrops += 1
    if backdrops:
        print(f"grscene: {backdrops} backdrop mesh(es) hidden", flush=True)
    if not los:
        raise RuntimeError(f"{usd_name} has no colliders; use the navigation variant")
    lo, hi = np.min(los, axis=0), np.max(his, axis=0)
    if max(hi[:2] - lo[:2]) > 100:
        raise RuntimeError(f"scene spans {hi - lo} m after scaling by {scale}; check the layer units")
    rugs = 0
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI) and any(f"/{c}/" in str(p.GetPath()) for c in drive_over):
            box = cache.ComputeWorldBound(p).ComputeAlignedBox()
            if not box.IsEmpty() and box.GetMax()[2] - lo[2] < drive_over_height:
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
                rugs += 1
    if rugs:
        print(f"grscene: {rugs} floor-level {'/'.join(drive_over)} colliders disabled (driven over)", flush=True)
    return GRScene(prim_path, lo, hi, scene_dir)


def build_occupancy(scene, cell=0.05, lidar_z=LIDAR_HEIGHT, ignore_prefix="/World/burger", cache=True,
                    low_object=0.005):
    """Scan the loaded scene with physics raycasts (after world.reset()). Cached next to the USD."""
    from pxr import Gf
    from omni.physx import get_physx_scene_query_interface

    path = scene.scene_dir / (f"occupancy_v4_{int(round(cell * 100))}cm_{int(round(lidar_z * 1000))}mm_"
                              f"{int(round(low_object * 1000))}mm.npz")
    if cache and path.exists():
        return Occupancy.load(path)
    pq = get_physx_scene_query_interface()
    lo, hi = scene.lo, scene.hi

    def cast(origin, direction, distance):
        res = pq.raycast_closest(Gf.Vec3f(*origin), Gf.Vec3f(*direction), distance)
        if res["hit"] and (str(res.get("rigidBody") or "").startswith(ignore_prefix)
                           or str(res.get("collision") or "").startswith(ignore_prefix)):
            return None
        return res["position"][2] if res["hit"] else None

    # Floor: downward hits over a coarse grid; the 10th percentile skips tables and beds.
    hits = [cast((x, y, hi[2] + 1.0), (0, 0, -1), hi[2] - lo[2] + 2.0)
            for x in np.linspace(lo[0], hi[0], 30) for y in np.linspace(lo[1], hi[1], 30)]
    hits = [z for z in hits if z is not None]
    floor = float(np.percentile(hits, 10)) if hits else float(lo[2])
    z = floor + lidar_z
    nx, ny = int((hi[0] - lo[0]) / cell) + 1, int((hi[1] - lo[1]) / cell) + 1
    grid = np.full((ny, nx), OUTSIDE, dtype=np.uint8)
    low = 0
    for j in range(ny):
        y = lo[1] + j * cell
        for i in range(nx):
            x = lo[0] + i * cell
            under = cast((x, y, z), (0, 0, -1), lidar_z + 0.5)
            if under is None:
                continue                           # no floor under this cell
            if any(cast((x, y, z), d, cell) is not None for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))):
                grid[j, i] = OCCUPIED              # seen by the lidar
            elif under > floor + low_object:       # shoes, steps: below the lidar plane, not drivable
                grid[j, i] = LOW                   # (rug colliders are disabled, so rays pass through rugs)
                low += 1
            else:
                grid[j, i] = FREE
    print(f"occupancy: floor z {floor:.3f} m, {low} cells on objects below the lidar plane", flush=True)
    occ = Occupancy((float(lo[0]), float(lo[1])), cell, grid)
    occ.floor = floor
    if cache:
        occ.save(path)
    return occ

"""Hierarchical navigation: grid path planning on top of the reactive DRL policy.

The policy was trained to reach a goal a few metres away in open space. In an
apartment the goal is often behind a wall, which a policy without memory or map
cannot route around. WaypointPlanner plans on the occupancy grid (8-connected
wavefront from the goal, so every free cell knows its distance to the goal) and
hands the policy a waypoint `lookahead` metres down the path instead of the
goal. The environment still judges success, timeout and collision against the
real goal; only the two goal entries of the observation change.

With cost_weight > 0 the planner is clearance-aware: the field is a cost-to-go from the goal
(Dijkstra, i.e. the A* cost model searched from the goal over the whole grid), with octile step
lengths (diagonal = sqrt 2) and each cell's cost raised by up to `cost_weight` times when it lies
within `cost_dist` of the inflated margin. Paths then keep to the middle of open space but still
use narrow passages. cost_weight = 0 keeps the original step-count wavefront.

`fallback` (set after construction) is a second planner, e.g. on the v4 grid, that takes over
wherever this planner's grid has no path from the robot to the goal: a finer grid can close a
passage the start/goal sampler (v4) still counts as open.

`open_passages` (radius, low radius) opens the narrow passages the margins close (Occupancy.inflated_open),
so a fine grid needs no fallback. With `los` the waypoint is the farthest point up to `lookahead` along the
path that the robot sees in a straight line without going deeper into the margin than it already is, and
within `final_within` the goal replaces the waypoint only when it is seen that way: the robot no longer
cuts corners through the margin towards a waypoint or goal behind a chair or table leg.

With `dyn_horizon` > 0 and moving obstacles given by set_obstacles() (ObstacleTracker), the waypoints follow
a field that charges the area each nearby obstacle will sweep in the next `dyn_horizon` s, so the policy is
led out of an obstacle's way instead of waiting in it (set_obstacles has the details).
"""
import heapq
import math

import numpy as np

from grscene import FREE, LOW, _dilate


class WaypointPlanner:
    def __init__(self, occupancy, inflate=0.18, lookahead=1.5, final_within=1.2, low_inflate=None,
                 stall_steps=30, stall_gain=0.10, recover_steps=20, recover_lookahead=1.5,
                 cost_weight=0.0, cost_dist=0.25, open_passages=None, los=False,
                 dyn_horizon=0.0, dyn_radius=0.40, dyn_cost=30.0, dyn_replan=3, dyn_range=3.0, dyn_dt=0.25,
                 dyn_timed=False, dyn_speed=0.2, dyn_window=(1.0, 0.7)):
        """`occupancy` is the raw grid; it is inflated by `inflate` (less than the sampler's
        0.25 m, so paths exist along furniture the policy can still pass) and LOW cells (below the
        lidar plane, invisible to the policy) by `low_inflate`.

        Stall recovery (step()): when the path length to the goal has not dropped by `stall_gain`
        for `stall_steps` control steps, the waypoint moves `recover_lookahead` ahead for
        `recover_steps` steps, and the stall is counted (the path_blocked signal)."""
        if open_passages:
            self.occ, self.passage = occupancy.inflated_open(inflate, low_inflate or inflate, *open_passages)
        else:
            self.occ = occupancy.inflated_split(inflate, low_inflate) if low_inflate else occupancy.inflated(inflate)
            self.passage = None
        self.stall_steps, self.stall_gain = stall_steps, stall_gain
        self.recover_steps, self.recover_lookahead = recover_steps, recover_lookahead
        self.reset_progress()
        self.free = self.occ.grid == FREE
        self.lookahead, self.final_within = lookahead, final_within
        self.free_cells = np.argwhere(self.free)
        self.goal = None
        self.dist = None
        self.replans = 0
        self.cost_weight = cost_weight
        self.fallback = None        # another WaypointPlanner (e.g. on the coarser v4 grid) used where this one has no path
        self.fallback_steps = 0
        self.penalty = self._clearance_penalty(occupancy, inflate, low_inflate or inflate, cost_dist) \
            if cost_weight > 0 else None
        self.los = los
        self.depth = self._margin_depth(occupancy, inflate, low_inflate or inflate) if los else None
        # moving obstacles (set_obstacles); dyn_horizon = 0 turns the dynamic field off
        self.dyn_horizon, self.dyn_radius, self.dyn_cost = dyn_horizon, dyn_radius, dyn_cost
        self.dyn_replan, self.dyn_range, self.dyn_dt = dyn_replan, dyn_range, dyn_dt
        self.dyn_timed, self.dyn_speed, self.dyn_window = dyn_timed, dyn_speed, tuple(dyn_window)
        n = int(math.ceil(dyn_radius / self.occ.cell))
        disc = np.array([(dr, dc) for dr in range(-n, n + 1) for dc in range(-n, n + 1) if dr * dr + dc * dc <= n * n])
        self._disc = (disc[:, 0], disc[:, 1])
        self.obstacles, self.goal_cell, self.dyn_replans = [], None, 0
        self._reset_dynamic()

    def _reset_dynamic(self):
        self.dist_dyn, self.dyn_extra, self.dyn_count = None, None, 0

    def set_obstacles(self, obstacles):
        """Moving obstacles as (x, y, vx, vy) in world coordinates. With dyn_horizon > 0, waypoint() follows a
        field that also charges `dyn_cost` per cell (fading to half over the horizon) on every cell within
        `dyn_radius` of where an obstacle within `dyn_range` of the robot will be in the next `dyn_horizon` s
        at its current velocity, recomputed every `dyn_replan` steps: the robot is led out of its way, and
        the line-of-sight check keeps waypoints from cutting through that swept area. Progress and stall
        detection (path_length) keep the static field."""
        self.obstacles = list(obstacles)

    def _dynamic_cost(self, obstacles, rx=None, ry=None):
        """Extra cost per cell for the moving obstacles. Untimed: every cell an obstacle sweeps within the
        horizon, more for the near future. Timed (dyn_timed): only the cells the robot, going straight at
        `dyn_speed`, would reach between dyn_window[0] s before and dyn_window[1] s after the obstacle, so
        passing behind an obstacle (or well ahead of it) stays cheap."""
        extra = np.zeros(self.free.shape)
        rows, cols = self.free.shape
        timed = self.dyn_timed and rx is not None
        for x, y, vx, vy in obstacles:
            for t in np.arange(0.0, self.dyn_horizon + 1e-9, self.dyn_dt):
                r, c = self.occ.to_cell(x + vx * t, y + vy * t)
                rr, cc = self._disc[0] + r, self._disc[1] + c
                ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
                rr, cc = rr[ok], cc[ok]
                if timed:
                    arrive = np.hypot(self.occ.origin[0] + cc * self.occ.cell - rx,
                                      self.occ.origin[1] + rr * self.occ.cell - ry) / self.dyn_speed
                    keep = (arrive >= t - self.dyn_window[0]) & (arrive <= t + self.dyn_window[1])
                    rr, cc = rr[keep], cc[keep]
                    cost = self.dyn_cost
                else:
                    cost = self.dyn_cost * (1.0 - 0.5 * t / self.dyn_horizon)
                extra[rr, cc] = np.maximum(extra[rr, cc], cost)
        return extra

    def _update_dynamic(self, rx, ry):
        near = [o for o in self.obstacles if math.hypot(o[0] - rx, o[1] - ry) < self.dyn_range]
        if not near or self.goal_cell is None:
            self._reset_dynamic()
            return
        if self.dist_dyn is None or self.dyn_count % self.dyn_replan == 0:
            self.dyn_extra = self._dynamic_cost(near, rx, ry)
            self.dist_dyn = self._cost_field(*self.goal_cell, extra=self.dyn_extra)
            self.dyn_replans += 1
        self.dyn_count += 1

    def _margin_depth(self, occupancy, inflate, low_inflate):
        """Per cell, how far (m) inside the planning margin it lies: 0 on FREE cells (opened passages
        included), at least 1 mm on every other cell, the full margin on obstacles."""
        cell, grid = occupancy.cell, occupancy.grid
        depth = np.zeros(grid.shape)
        for mask, margin, pad in (((grid != FREE) & (grid != LOW), inflate, True), (grid == LOW, low_inflate, False)):
            dist = np.full(grid.shape, np.inf)
            for n in range(int(math.ceil(margin / cell)), 0, -1):   # smallest disc that reaches the cell wins
                dist[_dilate(mask, n, pad)] = n * cell
            dist[mask] = 0.0
            depth = np.maximum(depth, np.clip(margin - dist, 0.0, None))
        depth[~self.free] = np.maximum(depth[~self.free], 1e-3)
        depth[self.free] = 0.0
        return depth

    def _depth_at(self, x, y):
        r, c = self.occ.to_cell(x, y)
        if 0 <= r < self.depth.shape[0] and 0 <= c < self.depth.shape[1]:
            return float(self.depth[r, c])
        return float(self.depth.max())

    def _los(self, x0, y0, x1, y1, max_depth, max_extra=None):
        """True when the straight line (x0, y0) -> (x1, y1) stays within `max_depth` of the margin's edge
        (and, with a dynamic field, never charges more for moving obstacles than `max_extra`)."""
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (self.occ.cell / 2)) + 1)
        t = np.linspace(0.0, 1.0, n)
        rows = np.rint((y0 + t * (y1 - y0) - self.occ.origin[1]) / self.occ.cell).astype(int)
        cols = np.rint((x0 + t * (x1 - x0) - self.occ.origin[0]) / self.occ.cell).astype(int)
        inside = (rows >= 0) & (rows < self.depth.shape[0]) & (cols >= 0) & (cols < self.depth.shape[1])
        if not inside.all():
            return False
        if max_extra is not None and (self.dyn_extra[rows, cols] > max_extra + 1e-9).any():
            return False
        return bool((self.depth[rows, cols] <= max_depth + 1e-9).all())

    @staticmethod
    def _clearance_penalty(occupancy, inflate, low_inflate, cost_dist):
        """Per cell 0..1: 1 at the inflated margin of an obstacle, falling linearly to 0 `cost_dist` beyond it."""
        cell, grid = occupancy.cell, occupancy.grid
        pen = np.zeros(grid.shape)
        for mask, margin in (((grid != FREE) & (grid != LOW), inflate), (grid == LOW, low_inflate)):
            n_max = int(math.ceil((margin + cost_dist) / cell))
            dist = np.full(grid.shape, np.inf)
            for n in range(n_max, 0, -1):          # smallest disc that reaches the cell wins
                dist[_dilate(mask, n, False)] = n * cell
            pen = np.maximum(pen, np.clip((margin + cost_dist - dist) / cost_dist, 0.0, 1.0))
        return pen

    def _cost_field(self, row, col, extra=None):
        """Cost-to-go from (row, col) over FREE cells: octile steps weighted by clearance (+ `extra` per cell)."""
        rows, cols = self.free.shape
        field = np.full(self.free.shape, np.inf)
        field[row, col] = 0.0
        w = 1.0 + (self.cost_weight * self.penalty if self.penalty is not None else 0.0)
        if extra is not None:
            w = w + extra
        heap = [(0.0, row, col)]
        steps = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > field[r, c]:
                continue
            for dr, dc, length in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and self.free[rr, cc]:
                    nd = d + length * 0.5 * (w[r, c] + w[rr, cc])
                    if nd < field[rr, cc]:
                        field[rr, cc] = nd
                        heapq.heappush(heap, (nd, rr, cc))
        return field

    def nearest_free(self, row, col):
        if 0 <= row < self.free.shape[0] and 0 <= col < self.free.shape[1] and self.free[row, col]:
            return row, col
        d = np.hypot(self.free_cells[:, 0] - row, self.free_cells[:, 1] - col)
        return tuple(int(v) for v in self.free_cells[int(np.argmin(d))])

    def set_goal(self, gx, gy):
        """Distance-to-goal field over the free cells (8-connected steps)."""
        if self.goal == (gx, gy) and self.dist is not None:
            return
        self.goal = (gx, gy)
        r, c = self.nearest_free(*self.occ.to_cell(gx, gy))
        self.goal_cell = (r, c)
        self._reset_dynamic()
        if self.cost_weight > 0:
            self.dist = self._cost_field(r, c)
        else:
            self.dist = self.occ.path_steps(r, c, max_steps=self.free.shape[0] + self.free.shape[1])
        self.replans += 1
        if self.fallback is not None:
            self.fallback.set_goal(gx, gy)

    def reset_progress(self):
        self.best, self.since, self.recovering, self.stalls = math.inf, 0, 0, 0
        if hasattr(self, "dyn_horizon"):
            self._reset_dynamic()

    def step(self, rx, ry):
        """waypoint() with stall recovery; returns (x, y, reachable, recovering)."""
        if self.dyn_horizon > 0:
            self._update_dynamic(rx, ry)
        length = self.path_length(rx, ry)
        if length < self.best - self.stall_gain:
            self.best, self.since = length, 0
        else:
            self.since += 1
        if self.recovering:
            self.recovering -= 1
        elif self.since >= self.stall_steps:
            self.recovering, self.since, self.best = self.recover_steps, 0, length
            self.stalls += 1
        x, y, ok = self.waypoint(rx, ry, self.recover_lookahead if self.recovering else None)
        return x, y, ok, bool(self.recovering)

    def waypoint(self, rx, ry, lookahead=None):
        """World (x, y) about `lookahead` m along the grid path from the robot, or the goal when close.

        Returns (x, y, reachable). Unreachable (robot cut off from the goal on the grid)
        falls back to the goal itself, so the policy behaves as without the planner.
        With `los`, see the module docstring: the farthest path point the robot sees within the
        lookahead, and the goal only when it is seen.
        """
        gx, gy = self.goal
        depth = self._depth_at(rx, ry) if self.los else None
        field = self.dist_dyn if getattr(self, "dist_dyn", None) is not None else self.dist
        extra = None                                       # moving obstacles: not deeper into their swept area
        if field is not self.dist and self.los:
            r0, c0 = self.occ.to_cell(rx, ry)
            inside = 0 <= r0 < self.free.shape[0] and 0 <= c0 < self.free.shape[1]
            extra = float(self.dyn_extra[r0, c0]) if inside else float(self.dyn_extra.max())
        if math.hypot(gx - rx, gy - ry) <= self.final_within and \
                (not self.los or self._los(rx, ry, gx, gy, depth, extra)):
            return gx, gy, True
        r, c = self.nearest_free(*self.occ.to_cell(rx, ry))
        if not np.isfinite(field[r, c]):
            if self.fallback is not None:
                self.fallback_steps += 1
                return self.fallback.waypoint(rx, ry, lookahead)
            return gx, gy, False
        travelled, cell = 0.0, self.occ.cell
        lookahead = self.lookahead if lookahead is None else lookahead
        cells = [(r, c)]                                   # the path from the robot's (nearest free) cell
        while travelled < lookahead:
            best, best_d = None, field[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < self.free.shape[0] and 0 <= cc < self.free.shape[1] \
                            and field[rr, cc] < best_d:
                        best, best_d = (rr, cc), field[rr, cc]
            if best is None:
                break
            travelled += cell * math.hypot(best[0] - r, best[1] - c)
            r, c = best
            cells.append(best)
            if best_d == 0:
                break
        if self.los:
            for pr, pc in reversed(cells):
                x, y = self.occ.to_world(pr, pc)
                if self._los(rx, ry, x, y, depth, extra):
                    return x, y, True
            r, c = cells[0]                                # nothing seen: back to the nearest free cell first
        x, y = self.occ.to_world(r, c)
        return x, y, True

    def path_length(self, rx, ry):
        """Length (m) of the grid path from (rx, ry) to the goal, following the distance field."""
        r, c = self.nearest_free(*self.occ.to_cell(rx, ry))
        if not np.isfinite(self.dist[r, c]):
            return self.fallback.path_length(rx, ry) if self.fallback is not None else math.inf
        length = 0.0
        while self.dist[r, c] > 0:
            best, best_d = None, self.dist[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < self.free.shape[0] and 0 <= cc < self.free.shape[1] \
                            and self.dist[rr, cc] < best_d:
                        best, best_d = (rr, cc), self.dist[rr, cc]
            if best is None:
                break
            length += self.occ.cell * math.hypot(best[0] - r, best[1] - c)
            r, c = best
        return length

    @staticmethod
    def observation_with(obs, rx, ry, heading, wx, wy, max_goal_distance):
        """Copy of the env observation with the goal distance/angle entries pointing at (wx, wy)."""
        obs = np.array(obs, dtype=np.float32, copy=True)
        d = math.hypot(wx - rx, wy - ry)
        a = math.atan2(wy - ry, wx - rx) - heading
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        n = len(obs) - 4                       # 40 scan values, then goal distance, goal angle, two previous actions
        obs[n] = min(max(d / max_goal_distance, 0.0), 1.0)
        obs[n + 1] = a / math.pi
        return obs


class ObstacleTracker:
    """(x, y, vx, vy) of the moving obstacles from the environment's pose samples ({index: (stamp, x, y)}).

    Velocities come from samples at least `min_dt` s apart; samples older than `max_age` s are dropped.
    In simulation the samples are the obstacles' true poses; a robot would feed tracked lidar clusters."""

    def __init__(self, min_dt=0.2, max_age=0.3):
        self.min_dt, self.max_age = min_dt, max_age
        self.reset()

    def reset(self):
        self.prev, self.vel = {}, {}

    def update(self, samples, now):
        out = []
        for i, (t, x, y) in samples.items():
            if now - t > self.max_age:
                continue
            p = self.prev.get(i)
            if p is None or t < p[0]:
                self.prev[i] = (t, x, y)
            elif t - p[0] >= self.min_dt:
                self.vel[i] = ((x - p[1]) / (t - p[0]), (y - p[2]) / (t - p[0]))
                self.prev[i] = (t, x, y)
            vx, vy = self.vel.get(i, (0.0, 0.0))
            out.append((x, y, vx, vy))
        return out


def obstacle_features(rx, ry, heading, obstacles, n=3, max_range=3.0, max_speed=0.6):
    """Policy input for the `n` nearest moving obstacles within `max_range`: per obstacle [present, x, y, vx, vy]
    in the robot frame (x ahead, y left), positions over max_range and velocities over max_speed, clipped to
    [-1, 1]; missing obstacles are all zeros. Goes in front of the observation (train.py --init-from then
    gives these inputs zero weights)."""
    c, s = math.cos(heading), math.sin(heading)
    near = sorted(((math.hypot(x - rx, y - ry), x, y, vx, vy) for x, y, vx, vy in obstacles), key=lambda o: o[0])
    out = np.zeros(5 * n, dtype=np.float32)
    k = 0
    for d, x, y, vx, vy in near:
        if k == n or d > max_range:
            break
        dx, dy = x - rx, y - ry
        out[5 * k:5 * k + 5] = (1.0, (c * dx + s * dy) / max_range, (-s * dx + c * dy) / max_range,
                                (c * vx + s * vy) / max_speed, (-s * vx + c * vy) / max_speed)
        k += 1
    return np.clip(out, -1.0, 1.0)


def rear_clearance(ranges, half_angle=60.0):
    """Smallest range (m) within `half_angle` degrees of straight behind; ranges go counterclockwise from ahead."""
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = np.arange(len(ranges)) * 360.0 / len(ranges)
    behind = np.abs((angles - 180.0 + 180.0) % 360.0 - 180.0) <= half_angle
    return float(ranges[behind].min())


def make_planner(scene_dir, plan_map="occupancy_v4", fallback_map=None, **kwargs):
    """WaypointPlanner on `scene_dir`/`plan_map`_*.npz, with a same-settings fallback planner on `fallback_map`."""
    import glob
    from grscene import Occupancy

    def load(prefix):
        return Occupancy.load(sorted(glob.glob(f"{scene_dir}/{prefix}_*.npz"))[0])
    planner = WaypointPlanner(load(plan_map), **kwargs)
    if fallback_map and fallback_map != plan_map:
        planner.fallback = WaypointPlanner(load(fallback_map), **kwargs)
    return planner


class ScanStack:
    """Older scans in front of the observation, so a reactive policy can tell moving obstacles from still ones.

    With frames=3, skip=2 the policy sees [scan(t-4), scan(t-2), observation(t)], 0.2 s apart at 0.1 s
    steps. The current observation stays last, so obs[-3] is still the waypoint bearing and a one-frame
    policy carries over (train.py --init-from gives the older scans zero weights). An episode's first
    observation fills the history."""

    def __init__(self, frames=1, skip=2, scan_size=40):
        self.frames, self.skip, self.scan_size = frames, skip, scan_size
        self.history = []

    def extra(self):
        return (self.frames - 1) * self.scan_size

    def reset(self, obs):
        self.history = [np.asarray(obs[:self.scan_size], dtype=np.float32)] * ((self.frames - 1) * self.skip + 1)
        return self.observation(obs)

    def push(self, obs):
        self.history = self.history[1:] + [np.asarray(obs[:self.scan_size], dtype=np.float32)]
        return self.observation(obs)

    def observation(self, obs):
        if self.frames == 1:
            return obs
        older = [self.history[k * self.skip] for k in range(self.frames - 1)]
        return np.concatenate(older + [np.asarray(obs, dtype=np.float32)]).astype(np.float32)

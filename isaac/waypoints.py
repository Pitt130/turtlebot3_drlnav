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
"""
import heapq
import math

import numpy as np

from grscene import FREE, LOW, _dilate


class WaypointPlanner:
    def __init__(self, occupancy, inflate=0.18, lookahead=1.5, final_within=1.2, low_inflate=None,
                 stall_steps=30, stall_gain=0.10, recover_steps=20, recover_lookahead=1.5,
                 cost_weight=0.0, cost_dist=0.25):
        """`occupancy` is the raw grid; it is inflated by `inflate` (less than the sampler's
        0.25 m, so paths exist along furniture the policy can still pass) and LOW cells (below the
        lidar plane, invisible to the policy) by `low_inflate`.

        Stall recovery (step()): when the path length to the goal has not dropped by `stall_gain`
        for `stall_steps` control steps, the waypoint moves `recover_lookahead` ahead for
        `recover_steps` steps, and the stall is counted (the path_blocked signal)."""
        self.occ = occupancy.inflated_split(inflate, low_inflate) if low_inflate else occupancy.inflated(inflate)
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

    def _cost_field(self, row, col):
        """Cost-to-go from (row, col) over FREE cells: octile steps weighted by clearance."""
        rows, cols = self.free.shape
        field = np.full(self.free.shape, np.inf)
        field[row, col] = 0.0
        w = 1.0 + self.cost_weight * self.penalty
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
        if self.cost_weight > 0:
            self.dist = self._cost_field(r, c)
        else:
            self.dist = self.occ.path_steps(r, c, max_steps=self.free.shape[0] + self.free.shape[1])
        self.replans += 1
        if self.fallback is not None:
            self.fallback.set_goal(gx, gy)

    def reset_progress(self):
        self.best, self.since, self.recovering, self.stalls = math.inf, 0, 0, 0

    def step(self, rx, ry):
        """waypoint() with stall recovery; returns (x, y, reachable, recovering)."""
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
        """
        gx, gy = self.goal
        if math.hypot(gx - rx, gy - ry) <= self.final_within:
            return gx, gy, True
        r, c = self.nearest_free(*self.occ.to_cell(rx, ry))
        if not np.isfinite(self.dist[r, c]):
            if self.fallback is not None:
                self.fallback_steps += 1
                return self.fallback.waypoint(rx, ry, lookahead)
            return gx, gy, False
        travelled, cell = 0.0, self.occ.cell
        lookahead = self.lookahead if lookahead is None else lookahead
        while travelled < lookahead:
            best, best_d = None, self.dist[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < self.free.shape[0] and 0 <= cc < self.free.shape[1] \
                            and self.dist[rr, cc] < best_d:
                        best, best_d = (rr, cc), self.dist[rr, cc]
            if best is None:
                break
            travelled += cell * math.hypot(best[0] - r, best[1] - c)
            r, c = best
            if best_d == 0:
                break
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

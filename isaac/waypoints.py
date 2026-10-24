"""Hierarchical navigation: grid path planning on top of the reactive DRL policy.

The policy was trained to reach a goal a few metres away in open space. In an
apartment the goal is often behind a wall, which a policy without memory or map
cannot route around. WaypointPlanner plans on the occupancy grid (8-connected
wavefront from the goal, so every free cell knows its distance to the goal) and
hands the policy a waypoint `lookahead` metres down the path instead of the
goal. The environment still judges success, timeout and collision against the
real goal; only the two goal entries of the observation change.
"""
import math

import numpy as np

from grscene import FREE


class WaypointPlanner:
    def __init__(self, occupancy, inflate=0.18, lookahead=1.5, final_within=1.2):
        """`occupancy` is the raw grid; it is inflated by `inflate` (less than the sampler's
        0.25 m, so paths exist along furniture the policy can still pass)."""
        self.occ = occupancy.inflated(inflate)
        self.free = self.occ.grid == FREE
        self.lookahead, self.final_within = lookahead, final_within
        self.free_cells = np.argwhere(self.free)
        self.goal = None
        self.dist = None
        self.replans = 0

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
        self.dist = self.occ.path_steps(r, c, max_steps=self.free.shape[0] + self.free.shape[1])
        self.replans += 1

    def waypoint(self, rx, ry):
        """World (x, y) about `lookahead` m along the grid path from the robot, or the goal when close.

        Returns (x, y, reachable). Unreachable (robot cut off from the goal on the grid)
        falls back to the goal itself, so the policy behaves as without the planner.
        """
        gx, gy = self.goal
        if math.hypot(gx - rx, gy - ry) <= self.final_within:
            return gx, gy, True
        r, c = self.nearest_free(*self.occ.to_cell(rx, ry))
        if not np.isfinite(self.dist[r, c]):
            return gx, gy, False
        travelled, cell = 0.0, self.occ.cell
        while travelled < self.lookahead:
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

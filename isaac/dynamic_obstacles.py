"""Moving obstacles for GRScenes apartments: people's legs or pets crossing the free space.

Each obstacle walks from one random point of the free space to another along the 8-connected grid
path of the start/goal sampler's grid (inflated for the robot, so it never cuts through furniture),
shortened where the straight line stays free, at a speed drawn per leg. It waits while its next move
would bring it closer than `yield_dist` (centre to centre) to the robot: people do not walk into a
robot, so a collision is the robot driving into them. With `yield_prob` < 1 each obstacle is drawn
per episode to yield or not; one that does not walks its route regardless of the robot. Every episode
draws its obstacle count, starts, routes and yielders from (seed, episode) only; just the waiting
depends on the robot. With `avoid` > 0 an obstacle also steers like a pedestrian: when keeping its
course for `avoid_horizon` seconds (robot at its current velocity) would bring the two centres closer
than `avoid`, it turns 30, 60 or 90 degrees, away from the robot first, onto a free course; only when
none is free does it fall back to waiting (if it yields) or walking on.

The cylinders are kinematic rigid bodies, so the lidar and the low ray layer see them, and the
runtime publishes their poses on /obstacle/odom as obstacleN::link (like gazebo_ros_p3d), from which
the environment counts a collision near one as COLL_OBST. Crowd has no Isaac dependency (tests).
"""
import math

import numpy as np

from grscene import FREE


class Crowd:
    def __init__(self, occupancy, count, min_count=None, radius=0.12, speed=(0.2, 0.5), yield_dist=0.45,
                 start_clearance=1.0, leg=(1.0, 4.0), give_way=2.0, yield_prob=1.0, avoid=0.0, avoid_horizon=2.0,
                 seed=0):
        """`occupancy` is the sampler's grid: its FREE cells are where an obstacle centre may go."""
        self.occ, self.count = occupancy, count
        self.min_count = count if min_count is None else min_count
        self.radius, self.speed, self.yield_dist, self.yield_prob = radius, speed, yield_dist, yield_prob
        self.avoid, self.avoid_horizon, self.robot_prev = avoid, avoid_horizon, None
        self.start_clearance, self.leg, self.give_way, self.seed = start_clearance, leg, give_way, seed or 0
        self.free = occupancy.grid == FREE
        self.cells = np.argwhere(self.free)
        self.xy = np.array([occupancy.to_world(r, c) for r, c in self.cells])
        self.active = 0
        self.pos = np.zeros((count, 2))
        self.vel = np.zeros((count, 2))
        self.routes = [[] for _ in range(count)]
        self.speeds = np.zeros(count)
        self.waited = np.zeros(count)
        self.yields = np.ones(count, dtype=bool)
        self.rng = np.random.default_rng(self.seed)

    def reset(self, episode, robot_xy):
        """New obstacle count, starts and routes for this episode, clear of the robot and of each other."""
        self.rng = np.random.default_rng([self.seed, episode])
        self.active = int(self.rng.integers(self.min_count, self.count + 1))
        self.vel[:], self.waited[:], self.robot_prev = 0.0, 0.0, None
        # own stream, so yield_prob = 1 keeps the starts and routes of runs made before it existed
        self.yields[:] = np.random.default_rng([self.seed, episode, 1]).random(self.count) < self.yield_prob
        far = np.hypot(self.xy[:, 0] - robot_xy[0], self.xy[:, 1] - robot_xy[1]) >= self.start_clearance
        for i in range(self.active):
            ok = far.copy()
            for j in range(i):
                ok &= np.hypot(self.xy[:, 0] - self.pos[j, 0], self.xy[:, 1] - self.pos[j, 1]) >= 2 * self.radius + 0.2
            if not ok.any():
                self.active = i
                break
            self.pos[i] = self.xy[self.rng.choice(np.flatnonzero(ok))]
            self._new_leg(i)

    def _cell(self, x, y):
        r, c = self.occ.to_cell(x, y)
        if 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1] and self.free[r, c]:
            return r, c
        d = np.hypot(self.cells[:, 0] - r, self.cells[:, 1] - c)
        return tuple(int(v) for v in self.cells[int(np.argmin(d))])

    def _free_at(self, xy):
        r, c = self.occ.to_cell(xy[0], xy[1])
        return 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1] and bool(self.free[r, c])

    def _miss(self, pos, v, robot, v_robot):
        """Closest centre distance within the horizon, both keeping their velocities; inf if not closing."""
        p, w = robot - pos, v - v_robot
        closing = float(p @ w)
        if closing <= 0.0:
            return math.inf
        t = min(closing / float(w @ w), self.avoid_horizon)
        return float(np.hypot(*(p - w * t)))

    def _steer(self, pos, nxt, robot, v_robot, dt):
        """A sidestep position for this tick, or None if the course is clear or no turn is free."""
        v = (nxt - pos) / dt
        speed = float(np.hypot(*v))
        if speed < 1e-9 or self._miss(pos, v, robot, v_robot) >= self.avoid:
            return None
        heading, p = math.atan2(v[1], v[0]), robot - pos
        away = -1.0 if v[0] * p[1] - v[1] * p[0] > 0 else 1.0       # robot on the left: turn right first
        for deg in (30, 60, 90):
            for sign in (away, -away):
                a = heading + sign * math.radians(deg)
                u = np.array([math.cos(a), math.sin(a)]) * speed
                if self._free_at(pos + u * dt) and self._miss(pos, u, robot, v_robot) >= self.avoid:
                    return pos + u * dt
        return None

    def _clear(self, a, b):
        """True if the segment a-b stays on FREE cells."""
        n = max(2, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.occ.cell / 2))) + 1)
        for t in np.linspace(0.0, 1.0, n):
            r, c = self.occ.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not (0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1] and self.free[r, c]):
                return False
        return True

    def _new_leg(self, i):
        r, c = self._cell(*self.pos[i])
        steps = self.occ.path_steps(r, c, int(self.leg[1] / self.occ.cell))
        reach = steps[self.cells[:, 0], self.cells[:, 1]] * self.occ.cell
        ok = (reach >= self.leg[0]) & (reach <= self.leg[1])
        if not ok.any():
            ok = np.isfinite(reach) & (reach > 0)
        if not ok.any():
            self.routes[i], self.speeds[i] = [], 0.0
            return
        cell = tuple(self.cells[self.rng.choice(np.flatnonzero(ok))])
        path = [cell]                                    # target back to the obstacle, down the wavefront
        while steps[cell] > 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = cell[0] + dr, cell[1] + dc
                    if (dr or dc) and 0 <= rr < steps.shape[0] and 0 <= cc < steps.shape[1] \
                            and steps[rr, cc] == steps[cell] - 1:
                        nxt = (rr, cc)
            cell = nxt
            path.append(cell)
        points = [self.occ.to_world(*p) for p in reversed(path)]
        route, k = [], 0
        while k < len(points) - 1:                       # skip ahead while the straight line stays free
            j = k + 1
            while j + 1 < len(points) and self._clear(points[k], points[j + 1]):
                j += 1
            route.append(points[j])
            k = j
        self.routes[i] = route
        self.speeds[i] = float(self.rng.uniform(*self.speed))

    def advance(self, dt, robot_xy):
        """Move every active obstacle `dt` seconds along its route, or let it wait for the robot; after
        `give_way` seconds of waiting it turns to a new leg (which may lead away from the robot)."""
        rx, ry = robot_xy
        robot = np.array(robot_xy, dtype=float)
        v_robot = (robot - self.robot_prev) / dt if self.robot_prev is not None else np.zeros(2)
        self.robot_prev = robot
        for i in range(self.active):
            if not self.routes[i]:
                self._new_leg(i)
            route, pos = self.routes[i], self.pos[i].copy()
            nxt, step, k = pos.copy(), self.speeds[i] * dt, 0
            while step > 0 and k < len(route):
                target = np.asarray(route[k], dtype=float)
                d = float(np.hypot(*(target - nxt)))
                if d <= step:
                    nxt, step, k = target, step - d, k + 1
                else:
                    nxt, step = nxt + (target - nxt) * (step / d), 0.0
            if self.avoid > 0:
                if not self._free_at(nxt):           # back from a sidestep, the straight line cuts a corner
                    self._new_leg(i)
                    nxt, k = pos, 0
                else:
                    side = self._steer(pos, nxt, robot, v_robot, dt)
                    if side is not None:
                        nxt, k = side, 0             # off the route for now; heads back to route[0] later
            d_next = math.hypot(nxt[0] - rx, nxt[1] - ry)
            if self.yields[i] and d_next < self.yield_dist and d_next < math.hypot(pos[0] - rx, pos[1] - ry):
                nxt, k = pos, 0
                self.waited[i] += dt
                if self.waited[i] >= self.give_way:
                    self.waited[i] = 0.0
                    self._new_leg(i)
            else:
                self.waited[i] = 0.0
            del self.routes[i][:k]
            self.vel[i] = (nxt - pos) / dt
            self.pos[i] = nxt


class IsaacCrowd:
    """Kinematic cylinders for a Crowd. Build before world.reset(); they wait under the scene, at
    `park_xy`, until attach() gives them a Crowd and reset() an episode (inactive ones stay parked)."""

    def __init__(self, world, count, radius, height, park_xy, root="/World/dyn_obstacles"):
        from isaacsim.core.prims import SingleXFormPrim
        from pxr import Gf, UsdGeom, UsdPhysics

        self.count, self.height, self.park_xy = count, height, park_xy
        self.crowd, self.floor = None, 0.0
        self.prims = []
        for i in range(count):
            path = f"{root}/obstacle{i + 1}"
            UsdGeom.Xform.Define(world.stage, path)
            cyl = UsdGeom.Cylinder.Define(world.stage, f"{path}/body")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.CreateAxisAttr("Z")
            cyl.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.1)])
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr(True)
            self.prims.append(SingleXFormPrim(path, name=f"dyn_obstacle{i + 1}"))
        self.park()

    def attach(self, crowd, floor):
        self.crowd, self.floor = crowd, floor

    @property
    def active(self):
        return self.crowd.active if self.crowd is not None else 0

    def park(self):
        """Every cylinder out of the scene, e.g. before the robot is placed on a new start."""
        import numpy as np
        for i, prim in enumerate(self.prims):
            prim.set_world_pose(position=np.array([self.park_xy[0] - 1.0 * i, self.park_xy[1], -10.0]))

    def apply(self):
        import numpy as np
        for i in range(self.active):      # the others stay where the last park() put them
            x, y = self.crowd.pos[i]
            self.prims[i].set_world_pose(position=np.array([x, y, self.floor + self.height / 2]))

    def reset(self, episode, robot_xy):
        self.park()
        self.crowd.reset(episode, robot_xy)
        self.apply()

    def advance(self, dt, robot_xy):
        if self.active:
            self.crowd.advance(dt, robot_xy)
            self.apply()

    def odometry(self):
        """(frame_id, x, y, vx, vy) of the active obstacles, as gazebo_ros_p3d names them."""
        return [(f"obstacle{i + 1}::link_obstacle{i + 1}", *map(float, self.crowd.pos[i]), *map(float, self.crowd.vel[i]))
                for i in range(self.active)]

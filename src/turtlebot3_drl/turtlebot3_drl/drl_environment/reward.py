from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

goal_dist_initial = 0

reward_function_internal = None

def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
    return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if succeed == SUCCESS:
            reward += 2500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            reward -= 2000
        return float(reward)

# Indoor variant of A for furnished apartments (GRScenes). A was tuned for an open
# 4 m arena: it charges -20 per step within 0.22 m of anything, which in a 0.8 m
# doorway means both walls at once, so passing a door costs as much as a crash;
# and a timeout carries no terminal cost, so stalling until the clock runs out is
# cheaper than trying. B keeps A's shaping but
#   - charges the proximity penalty only inside OBSTACLE_PENALTY_DIST (0.15 m,
#     2 cm above the collision threshold) and scales it to -10,
#   - adds a per-step progress term, 50 * (previous goal distance - current),
#     about +1.1 per step at full speed towards the goal (A's r_distance only
#     compares with the episode's initial distance and saturates),
#   - ends a timeout with -500 and a tumble with -1000.
OBSTACLE_PENALTY_DIST = 0.15
_prev_goal_dist = None


def get_reward_B(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
    global _prev_goal_dist
    r_yaw = -1 * abs(goal_angle)
    r_vangular = -1 * (action_angular ** 2)
    r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1
    r_obstacle = -10 if min_obstacle_dist < OBSTACLE_PENALTY_DIST else 0
    r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)
    progress = 0.0 if _prev_goal_dist is None else (_prev_goal_dist - goal_dist)
    _prev_goal_dist = goal_dist
    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular + 50 * progress - 1
    if succeed == SUCCESS:
        reward += 2500
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward -= 2000
    elif succeed == TIMEOUT:
        reward -= 500
    elif succeed == TUMBLE:
        reward -= 1000
    return float(reward)


# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial, _prev_goal_dist
    goal_dist_initial = init_distance_to_goal
    _prev_goal_dist = init_distance_to_goal      # B's progress term starts from the episode's first distance

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")

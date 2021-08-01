import numpy as np
from collections import OrderedDict

def is_accelerating():
    pass

def is_deccelerating():
    pass

def is_turning_right(agent_trajectory_dict):
    traj1 = agent_trajectory_dict

    steering_history = np.concatenate([np.array([traj1['current_steering']]), traj1['past_steering']])
    chosen_history_len = min(4, steering_history.size)

    recent_steering_history = steering_history[-chosen_history_len:]
    if recent_steering_history.mean() < -0.1:
        return True
    else:
        return False
    

def is_turning_left(agent_trajectory_dict):
    traj1 = agent_trajectory_dict

    steering_history = np.concatenate([np.array([traj1['current_steering']]), traj1['past_steering']])
    chosen_history_len = min(4, steering_history.size)

    recent_steering_history = steering_history[-chosen_history_len:]

    if recent_steering_history.mean() > 0.1:
        return True
    else:
        return False


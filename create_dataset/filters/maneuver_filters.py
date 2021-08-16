import numpy as np
from collections import OrderedDict
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction.helper import angle_of_rotation

def ego_maneuver_filter(r):
    #### Steering ####
    # use average of past 5 timesteps
    # past_steering = [s[-1] for s in r.past_agent_steering[-4:] if len(r.past_agent_steering) > 0]
    # ego_steering_history = [0]*4 + past_steering + [r.current_agent_steering[-1]]

    ego_steering_history = r.past_agent_steering[-4:]

    if np.array(ego_steering_history).mean() < -0.1:
        return 'turn_left'
    elif np.array(ego_steering_history).mean() > 0.1:
        return 'turn_right'
    else:
        return 'straight'

def ado_maneuver_filter(r):
    ado_heading = []
    for q in r.past_agent_quat:
        #### convert from global quat to local steering ####
        yaw = Quaternion(q)
        yaw = quaternion_yaw(yaw)
        ado_heading.append(-angle_of_rotation(yaw))

    ado_heading = np.array(ado_heading)
    steering = (ado_heading[1:] - ado_heading[:-1])/0.5

    if steering.mean() < -0.1:
        return 'turn_left'
    elif steering.mean() > 0.1:
        return 'turn_right'
    else:
        return 'straight'

def maneuver_filter(scene_df):
    current_agent_maneuvers = []
    past_agent_maneuvers = []
    future_agent_maneuvers = []
    
    for i, r in scene_df.iterrows():
        obs_steps = r.past_agent_pos.shape[0]
        pred_steps = r.future_agent_pos.shape[0]

        # current_maneuvers_i = []
        # past_maneuvers_i = []
        # future_maneuvers_i = []
        # for t in range(obs_steps+pred_steps+1):
        #     if r.agent_token == 'ego':
        #         maneuvers = ego_maneuver_filter(r)
        #     else:
        #         maneuvers = ado_maneuver_filter(r)

        #     if t == obs_steps:
        #         current_maneuvers_i.append(maneuvers)
        #     if t < obs_steps:
        #         past_maneuvers_i.append(maneuvers)
        #     if t > obs_steps:
        #         future_maneuvers_i.append(maneuvers)

        # current_agent_maneuvers.append(current_maneuvers_i)
        # past_agent_maneuvers.append(past_maneuvers_i)
        # future_agent_maneuvers.append(future_maneuvers_i)
        
        if r.agent_token == 'ego':
            maneuvers = ego_maneuver_filter(r)
        else:
            maneuvers = ado_maneuver_filter(r)

        current_agent_maneuvers.append(maneuvers)
        past_agent_maneuvers.append(maneuvers)
        future_agent_maneuvers.append(maneuvers)
    
    scene_df['current_agent_maneuvers'] = current_agent_maneuvers
    scene_df['past_agent_maneuvers'] = past_agent_maneuvers
    scene_df['future_agent_maneuvers'] = future_agent_maneuvers

    return scene_df

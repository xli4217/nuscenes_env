import numpy as np
from collections import OrderedDict


def ego_maneuver_filter(scene_df):
    ego_maneuver_list = []
    for i, r in scene_df.iterrows():
        current_ego_maneuvers = []

        #### Steering ####
        # use average of past 5 timesteps
        past_steering = [s[-1] for s in r.past_ego_steering[-4:] if len(r.past_ego_steering) > 0]
    
        ego_steering_history = [0]*4 + past_steering + [r.current_ego_steering[-1]]

        if np.array(ego_steering_history).mean() < -0.1:
            current_ego_maneuvers.append('turn_left')
        elif np.array(ego_steering_history).mean() > 0.1:
            current_ego_maneuvers.append('turn_right')

        ego_maneuver_list.append(current_ego_maneuvers)


    scene_df['current_ego_maneuvers'] = ego_maneuver_list
    return scene_df

def ado_maneuver_filter(scene_df):
    return scene_df

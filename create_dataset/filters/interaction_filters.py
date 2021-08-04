import numpy as np
from collections import OrderedDict
from utils.utils import process_to_len
from shapely.geometry import LineString, Polygon

#######################
# Filtering Functions #
#######################
def is_follow(agent1_traj, agent2_traj):
    """ Determines is the first agent is following the second agent

    :param agent_trajectories: maps agent_id to a dictionary containing its past traj, current position, and future traj 
    :param map_elements: maps agent_id to a dictionary containing road elements specific to that agent i.e. closest lane, etc
    :returns: True if first agent is following second agent, False otherwise

    """
    
    traj1, traj2 = agent1_traj, agent2_traj

    max_len = max(traj1.shape[0], traj2.shape[0]) + 3
    traj1 = process_to_len(traj1, max_len, name='traj1', dim=0, before_or_after='after', mode='edge')
    traj2 = process_to_len(traj2, max_len, name='traj2', dim=0, before_or_after='after', mode='edge')

    v1 = traj2[0] - traj1[0] + 1e-4
    v1 = v1 / np.linalg.norm(v1)

    v2 = traj1[3] - traj1[0] + 1e-4
    v2 = v2 / np.linalg.norm(v2)

    v3 = traj2[3] - traj2[0] + 1e-4
    v3 = v3 / np.linalg.norm(v3)

    angle1 = np.rad2deg(np.arccos(np.dot(v1, v2)))
    angle2 = np.rad2deg(np.arccos(np.dot(v2, v3)))

    d = np.linalg.norm(traj1 - traj2[0], axis=-1)
    angle_th = 15
    if d.min() < 5. and -angle_th < angle1 < angle_th and -angle_th < angle2 < angle_th:
        return True
    else:
        return False

def get_interactions(row, sample_df, interaction_name, interaction_func, ego_or_ado='ego'):
    interactions = []

    if ego_or_ado == 'ego':
        neighbors = row.current_ego_neighbors
        traj1 = np.array([row.current_ego_pos] + row.future_ego_pos)[:,:2]
    elif ego_or_ado == 'ado':
        neighbors = row.current_instance_neighbors
        traj1 = np.array([row.current_instance_pos] + row.future_instance_pos)[:,:2]

    for instance_token, dist in zip(neighbors[0], neighbors[1]):
        if dist < 40:
            if instance_token == 'ego':
                row2 = row
                traj2 = np.array([row2.current_ego_pos] + row2.future_ego_pos)[:,:2]
            else:
                row2 = sample_df.loc[sample_df.instance_token==instance_token]
                traj2 = np.array(row2.current_instance_pos.tolist() + row2.future_instance_pos.tolist()[0])[:,:2]

            if interaction_func(traj1, traj2):
                interactions.append((interaction_name, instance_token))
    return interactions
                
###########
# Filters #
###########

def lead_follow_filter(scene_df):
    ego_interactions = []
    instance_interactions = []

    sample_based_df = scene_df.set_index(['sample_idx'])
    for sample_idx, sample_df in sample_based_df.groupby(level='sample_idx'):
        current_ego_interactions = get_interactions(sample_df.iloc[0], sample_df, interaction_name='follows', interaction_func=is_follow, ego_or_ado='ego')
        ego_interactions += [current_ego_interactions] * sample_df.shape[0]

        # ado interactions #
        for i, r in sample_df.iterrows():
            current_ado_interactions = get_interactions(r, sample_df, interaction_name='follows', interaction_func=is_follow, ego_or_ado='ado')
            instance_interactions.append(current_ado_interactions)
            
    scene_df['current_ego_interactions'] = ego_interactions
    scene_df['current_instance_interactions'] = instance_interactions
    return scene_df

def yield_filter(scene_df):
    return scene_df

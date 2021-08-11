import numpy as np
from collections import OrderedDict
from utils.utils import process_to_len, assert_type_and_shape
from shapely.geometry import LineString, Polygon

#######################
# Filtering Functions #
#######################
def is_follow(agent1_traj, speed1=None, agent2_traj=None, speed2=None, params={}):
    """ Determines is the first agent is following the second agent

    :param agent_trajectories: maps agent_id to a dictionary containing its past traj, current position, and future traj 
    :param map_elements: maps agent_id to a dictionary containing road elements specific to that agent i.e. closest lane, etc
    :returns: True if first agent is following second agent, False otherwise

    """
    if agent1_traj.shape[0] < 3:
        return False
    
    traj1, traj2 = agent1_traj, agent2_traj

    # max_len = max(traj1.shape[0], traj2.shape[0]) + 3
    # traj1 = process_to_len(traj1, max_len, name='traj1', dim=0, before_or_after='after', mode='edge')
    # traj2 = process_to_len(traj2, max_len, name='traj2', dim=0, before_or_after='after', mode='edge')

    v1 = traj2[0] - traj1[0] + 1e-4
    v1 = v1 / np.linalg.norm(v1)

    v2 = traj1[2] - traj1[0] + 1e-4
    v2 = v2 / np.linalg.norm(v2)

    v3 = traj2[2] - traj2[0] + 1e-4
    v3 = v3 / np.linalg.norm(v3)

    angle1 = np.rad2deg(np.arccos(np.dot(v1, v2)))
    angle2 = np.rad2deg(np.arccos(np.dot(v2, v3)))

    d = np.linalg.norm(traj1 - traj2[0], axis=-1)
    angle_th = 15
    if d.min() < 5. and -angle_th < angle1 < angle_th and -angle_th < angle2 < angle_th:
        return True
    else:
        return False

def yields(agent1_traj, speed1=None, agent2_traj=None, speed2=None, params={}):
    """
    if agent1 and agent2 start on different lanes and end up in the same lane &&
    agent1 currently has low velocity
    """
    if agent1_traj.shape[0] < 3:
        return False
    
    traj1, traj2 = agent1_traj, agent2_traj
    v1 = speed1
    v2 = speed2

    v1_dir = (traj1[2] - traj1[0]) / np.linalg.norm(traj1[2] - traj1[0])
    v2_dir = (traj2[2] - traj2[0]) / np.linalg.norm(traj2[2] - traj2[0])

    v_angle = np.arccos(np.dot(v1_dir, v2_dir))

    # if "2f44" in list(agent_trajectories.keys())[0] and "b715" in list(agent_trajectories.keys())[1]:
        
    if v1[0] < 3 and np.rad2deg(v_angle) > 40: # agent1 is slowed down
        sl1 = LineString(traj1.tolist())
        plyg1 = Polygon(sl1)

        sl2 = LineString(traj2.tolist())
        plyg2 = Polygon(sl2)

        if plyg1.intersects(plyg2):
            dist = sl1.boundary[0].distance(sl2.boundary[0])

            if dist > 3:
                return True

    return False

    
def yields1(agent1_traj, speed1=None, agent2_traj=None, speed2=None, params={}):
    """Returns True if agent1 and agent2 start on different paths, end up in the same path, and agent2 gets first to the joint part of the path.
    """

    if agent1_traj.shape[0] < 3:
        return False
    
    def compute_trajectory_stats(traj, speed, dilation):
        """Compute trajectory stats needed for yielding"""
        #TOt
        positions = traj[:, :2]
        spd1 = speed
        t = 0.5*np.arange(traj.shape[0])
        sl = LineString(positions.tolist()).buffer(dilation)
        return positions, spd1, t, sl

    yielding_dilation_radius = params["yielding_dilation_radius"]

    positions1, spd1, t1, sl1 = compute_trajectory_stats(agent1_traj, speed1, yielding_dilation_radius)

    positions2, spd2, t2, sl2 = compute_trajectory_stats(agent2_traj, speed2, yielding_dilation_radius)

    intersection = sl1.intersection(sl2)
    idxs1 = find_points_in_region(positions1, intersection)
    idxs2 = find_points_in_region(positions2, intersection)
    if len(idxs1) == 0 or len(idxs2) == 0:
        return False

    # The trajectories upto the intersection.
    positions1b = positions1[: idxs1[0], :]
    positions2b = positions2[: idxs2[0], :]
    if positions1b.shape[0] < 2 or positions2b.shape[0] < 2:
        return False

    sl1b = LineString(positions1b[:, :2]).buffer(yielding_dilation_radius)
    sl2b = LineString(positions2b[:, :2]).buffer(yielding_dilation_radius)

    # The regions that cross eachother
    sl1_only = (sl1b.difference(sl2)).buffer(-yielding_dilation_radius / 2.0)
    sl2_only = (sl2b.difference(sl1)).buffer(-yielding_dilation_radius / 2.0)
    # Skip if no overlaps in the trajectory.
    if len(idxs1) == 0 or len(idxs2) == 0:
        return False
    
    idx1_m1 = max(0, idxs1[0] - params["yielding_dt"])
    if (sl1_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 1
        and sl2_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 2
        and t1[idxs1[0]] - t2[idxs2[0]] > params["yielding_time_gap"]  # agent 1 is before agent 2
        and spd1[idx1_m1] - spd1[idxs1[0]] > -0.5  # non-increasing speed.
        and Point(positions2b[0, :]).distance(sl1_only)
        > params["yielding_initial_distance"]  # initial point for 2 is far from trajectory 1
    ):
        return True
    return False

    
def get_interactions(row, sample_df, interaction_name, interaction_func, ego_or_ado='ego', params={}):
    interactions = []

    if ego_or_ado == 'ego':
        neighbors = row.current_ego_neighbors
        traj1 = np.array([row.current_ego_pos] + row.future_ego_pos)[:,:2]
        sp1 = np.array([row.current_ego_speed] + row.future_ego_speed)[:,:2]
    elif ego_or_ado == 'ado':
        neighbors = row.current_instance_neighbors
        traj1 = np.array([row.current_instance_pos] + row.future_instance_pos)[:,:2]
        sp1 = np.array([row.current_instance_speed] + row.future_instance_speed)[:,:2]

    for instance_token, dist in zip(neighbors[0], neighbors[1]):
        if dist < 40:
            if instance_token == 'ego':
                row2 = row
                traj2 = np.array([row2.current_ego_pos] + row2.future_ego_pos)[:,:2]
                sp2 = np.array([row2.current_ego_speed] + row2.future_ego_speed)[:,:2]
            else:
                row2 = sample_df.loc[sample_df.instance_token==instance_token]
                traj2 = np.array(row2.current_instance_pos.tolist() + row2.future_instance_pos.tolist()[0])[:,:2]
                sp2 = np.array(row2.current_instance_speed.tolist() + row2.future_instance_speed.tolist()[0])[:,:2]

            if interaction_func(traj1, speed1,  traj2, speed2, params):
                interactions.append((interaction_name, instance_token))
    return interactions
                
###########
# Filters #
###########

def interaction_filter(scene_df):
    current_interactions = []
    past_interactions = []
    future_interactions = []

    for i, r in scene_df.iterrows():
        obs_steps = r.past_agent_pos.shape[0]
        pred_steps = r.future_agent_pos.shape[0]

        if r.past_agent_speed.ndim == 2:
            r.current_agent_speed = r.current_agent_speed[0]
            r.past_agent_speed = r.past_agent_speed[:,0]
            r.future_agent_speed = r.future_agent_speed[:,0]

        agents_traj_dict = {
            r.agent_token: {
                'pos': np.vstack([r.past_agent_pos, r.current_agent_pos[np.newaxis], r.future_agent_pos]),
                'speed': np.concatenate([r.past_agent_speed, np.array([r.current_agent_speed]), r.future_agent_speed])
            }
        }

        for i, n_token in enumerate(r.current_neighbor_tokens):
            agents_traj_dict[n_token] =  {'pos': np.vstack([r.past_neighbor_pos[i], r.current_neighbor_pos[i][np.newaxis], r.future_neighbor_pos[i]]),
                                          'speed': np.concatenate([r.past_neighbor_speed[i], np.array([r.current_neighbor_speed[i]]), r.future_neighbor_speed[i]])
            }

        all_tokens = [r.agent_token] + r.current_neighbor_tokens
        
        current_interactions_i = []
        past_interactions_i = []
        future_interactions_i = []
        # loop through all times #
        for t in range(obs_steps+pred_steps+1):
            interactions = []
            # loop through any two agents in the neighborhood #
            for i , n1_token in enumerate(all_tokens):
                for j, n2_token in enumerate(all_tokens):
                    if i != j:
                        a1 = agents_traj_dict[n1_token]
                        a2 = agents_traj_dict[n2_token]
                        # loop through all interaction types
                        for interaction_name in ['follows', 'yields']:
                            if interaction_name == 'follows':
                                interaction_func = is_follow
                                params = {}
                            elif interaction_name == 'yields':
                                interaction_func = yields
                                params = {
                                    'yielding_dilation_radius': 10,
                                    'yielding_prefix': 4,
                                    'yielding_dt': 0.5,
                                    'yielding_time_gap': 4,
                                    'yielding_initial_distance': 4
                                }
                            else:
                                raise ValueError()

                            has_interaction = interaction_func(a1['pos'][t:], a1['speed'][t:],
                                                               a2['pos'][t:], a2['speed'][t:],
                                                               params)
                            if has_interaction:
                                interactions.append((n1_token, interaction_name, n2_token))

            if t == obs_steps:
                current_interactions_i.append(interactions)
            if t < obs_steps:
                past_interactions_i.append(interactions)
            if t > obs_steps:
                future_interactions_i.append(interactions)

        current_interactions.append(current_interactions_i)
        past_interactions.append(past_interactions_i)
        future_interactions.append(future_interactions_i)

    scene_df['current_interactions'] = current_interactions
    scene_df['past_interactions'] = past_interactions
    scene_df['future_interactions'] = future_interactions

    return scene_df
                            
def interaction_filter1(scene_df, interaction_name):
    ego_interactions = []
    instance_interactions = []

    sample_based_df = scene_df.set_index(['sample_idx'])
    for sample_idx, sample_df in sample_based_df.groupby(level='sample_idx'):
        current_ego_interactions = get_interactions(sample_df.iloc[0], sample_df, interaction_name='follows', interaction_func=is_follow, ego_or_ado='ego')
        ego_interactions += [current_ego_interactions] * sample_df.shape[0]

        # ado interactions #
        for i, r in sample_df.iterrows():
            if interaction_name == 'follows':
                interaction_func = is_follow
                params = {}
            elif interaction_name == 'yields':
                interaction_func = yields
                params = {
                    'yielding_diation_radius': 10,
                    'yielding_prefix': 4,
                    'yielding_dt': 0.5,
                    'yielding_time_gap': 4,
                    'yielding_initial_distance': 4
                }
            else:
                raise ValueError()
            current_ado_interactions = get_interactions(r, sample_df, interaction_name=interaction_name, interaction_func=interaction_func, ego_or_ado='ado', params=params)
            instance_interactions.append(current_ado_interactions)
            
    scene_df['current_ego_interactions'] = ego_interactions
    scene_df['current_instance_interactions'] = instance_interactions
    return scene_df


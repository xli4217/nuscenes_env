import numpy as np
from collections import OrderedDict
from utils.utils import process_to_len
from shapely.geometry import LineString, Polygon

def pad_trajs(agent_trajectories: OrderedDict):
    traj1 = list(agent_trajectories.values())[0]
    traj2 = list(agent_trajectories.values())[1]

    if traj1['future'].ndim == 1 or traj2['future'].ndim == 1:
        return None

    traj1 = np.vstack([traj1['current'][np.newaxis,:], traj1['future']])
    traj2 = np.vstack([traj2['current'][np.newaxis,:], traj2['future']])

    traj_len = max(4, min(traj1.shape[0], traj2.shape[0]))
    traj1 = process_to_len(traj1, traj_len)[:,:2]
    traj2 = process_to_len(traj2, traj_len)[:,:2]


    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"
    
    return traj1, traj2

def is_follow(agent_trajectories: OrderedDict, map_elements: OrderedDict):
    """ Determines is the first agent is following the second agent

    :param agent_trajectories: maps agent_id to a dictionary containing its past traj, current position, and future traj 
    :param map_elements: maps agent_id to a dictionary containing road elements specific to that agent i.e. closest lane, etc
    :returns: True if first agent is following second agent, False otherwise

    """
    out = pad_trajs(agent_trajectories)

    if out is None:
        return False
    
    traj1, traj2 = out
    
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

def is_yielding(agent_trajectories: OrderedDict, map_elements: OrderedDict):
    """
    if agent1 and agent2 start on different lanes and end up in the same lane &&
    agent1 currently has low velocity
    """
    out = pad_trajs(agent_trajectories)
    if out is None:
        return False

    traj1, traj2 = out
    v1 = list(agent_trajectories.values())[0]['vel']
    v2 = list(agent_trajectories.values())[1]['vel']

    v1_dir = (traj1[1] - traj1[0]) / np.linalg.norm(traj1[1] - traj1[0])
    v2_dir = (traj2[1] - traj2[0]) / np.linalg.norm(traj2[1] - traj2[0])

    v_angle = np.arccos(np.dot(v1_dir, v2_dir))

    # if "2f44" in list(agent_trajectories.keys())[0] and "b715" in list(agent_trajectories.keys())[1]:

    #     import ipdb; ipdb.set_trace()
        
    if v1 < 3 and np.rad2deg(v_angle) > 40: # agent1 is slowed down
        sl1 = LineString(traj1.tolist())
        plyg1 = Polygon(sl1)

        sl2 = LineString(traj2.tolist())
        plyg2 = Polygon(sl2)

        if plyg1.intersects(plyg2):
            dist = sl1.boundary[0].distance(sl2.boundary[0])

            if dist > 3:
                return True

    return False
    
    
def is_overtaking(agent_trajectories: OrderedDict, map_elements: OrderedDict):
    '''
    returns true if traj1 is overtaking traj2
    '''

    

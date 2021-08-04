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
    
def yield_filter(agent_trajectories: OrderedDict, map_elements: OrderedDict, params: dict) -> list:
    """If agent1 and agent2 start on different paths, end up in the same path, and agent2 gets first to the joint part of the path.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        Map from agent id to the trajectory of the agent.
    map_elements: OrderedDict
        Map information
    params: dict
        Parameters

    Returns
    -------
    token_list: list
        A list of tokens detected ("yield").
    """
    if len(list(agent_trajectories.values())) < 2:
        return []

    def compute_trajectory_stats(traj, dilation):
        """Compute trajectory stats needed for yielding"""
        #TOt
        positions = traj[:, :2]
        v = positions[1:, :] - positions[:-1, :]
        v = np.concatenate((v, v[-2:-1, :]), axis=0)
        spd1 = (v ** 2).sum(1)
        t = traj[:, 2]
        sl = LineString(positions.tolist()).buffer(dilation)
        return positions, spd1, t, sl

    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    yielding_dilation_radius = params["yielding_dilation_radius"]

    positions1, spd1, t1, sl1 = compute_trajectory_stats(target_agent_traj, yielding_dilation_radius)

    tokens = []
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        other_agent_traj = agent_trajectories[other_agent_id]
        positions2, spd2, t2, sl2 = compute_trajectory_stats(other_agent_traj, yielding_dilation_radius)
        intersection = sl1.intersection(sl2)
        idxs1 = find_points_in_region(positions1, intersection)
        idxs2 = find_points_in_region(positions2, intersection)
        if len(idxs1) == 0 or len(idxs2) == 0:
            continue

        # The trajectories upto the intersection.
        positions1b = positions1[: idxs1[0], :]
        positions2b = positions2[: idxs2[0], :]
        if positions1b.shape[0] < 2 or positions2b.shape[0] < 2:
            continue

        sl1b = LineString(positions1b[:, :2]).buffer(yielding_dilation_radius)
        sl2b = LineString(positions2b[:, :2]).buffer(yielding_dilation_radius)

        # The regions that cross eachother
        sl1_only = (sl1b.difference(sl2)).buffer(-yielding_dilation_radius / 2.0)
        sl2_only = (sl2b.difference(sl1)).buffer(-yielding_dilation_radius / 2.0)
        # Skip if no overlaps in the trajectory.
        if len(idxs1) == 0 or len(idxs2) == 0:
            continue
        idx1_m1 = max(0, idxs1[0] - params["yielding_dt"])
        if (
            sl1_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 1
            and sl2_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 2
            and t1[idxs1[0]] - t2[idxs2[0]] > params["yielding_time_gap"]  # agent 1 is before agent 2
            and spd1[idx1_m1] - spd1[idxs1[0]] > -0.5  # non-increasing speed.
            and Point(positions2b[0, :]).distance(sl1_only)
            > params["yielding_initial_distance"]  # initial point for 2 is far from trajectory 1
        ):
            tokens.append(create_duration_label("Yield", t2[idxs2[0]], t2[idxs2[-1]], subject=other_agent_id))
    return tokens
    

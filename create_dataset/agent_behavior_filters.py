import numpy as np


def agent_interactions(traj1, road_objects1, traj2, road_objects2):
    interactions =  []

    if is_follow(traj1, road_objects1, traj2, road_objects2):
        interactions.append('follow')

    return interactions

def road_interactions(road_objects):
    return on_road_objects(road_objects)

##################
# Between Agents #
##################
def is_follow(traj1, road_objects1, traj2, road_objects2):
    '''
    returns true if traj1 is following traj2
    '''
    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"


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

def is_yielding(traj_1, traj2):
    '''
    returns true if traj1 is yielding to traj2
    '''
    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"
    pass

def is_overtaking(traj1, traj2):
    '''
    returns true if traj1 is overtaking traj2
    '''
    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"
    pass

######################
# With Road Elements #
######################
def on_road_objects(road_objects, road_element_type=['intersection']):
    on_road_objects = []
    for k, v in road_objects.items():
        if v != "" and k in road_element_type:
            on_road_objects.append(k)
    return on_road_objects

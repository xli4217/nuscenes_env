import numpy as np
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion import arcline_path_utils
from pyquaternion import Quaternion
import tqdm
from typing import Tuple
from utils.utils import convert_global_coords_to_local, convert_local_coords_to_global

def get_discretized_future_lanes(ego_pos, ego_quat, lane_record, nusc_map, frame='global', ego_speed=2):

    lane_discretized_pos_global = np.array(arcline_path_utils.discretize_lane(lane_record, resolution_meters=ego_speed))
    #if lane_discretized_pos_global.ndim == 2:
    lane_discretized_pos_global = lane_discretized_pos_global[:,:2]
    # else:
    #     # here !!!!!!!!!
    #     if frame == 'global':
    #         lane_discretized_poses = np.repeat()

    #     elif frame == 'local':
    #         lane_discretized_poses = lane_discretized_poses_local

        
    lane_discretized_poses_local = convert_global_coords_to_local(lane_discretized_pos_global, np.array(ego_pos), np.array(ego_quat))

    # get part of current lane that's in front of the ego
    lane_discretized_poses_local_forward_idx = np.argmax(lane_discretized_poses_local[:,1] > 0)
    lane_discretized_poses_local = lane_discretized_poses_local[lane_discretized_poses_local_forward_idx:, :]

    if frame == 'global':
        lane_discretized_poses = convert_local_coords_to_global(lane_discretized_poses_local, ego_pos, ego_quat)

    elif frame == 'local':
        lane_discretized_poses = lane_discretized_poses_local

    return lane_discretized_poses


def get_future_lanes(nusc_map, ego_pos, ego_quat, frame='global', ego_speed=2):
    ## lanes ##
    # closest lane
    closest_lane = nusc_map.get_closest_lane(ego_pos[0], ego_pos[1], radius=2)
    lane_record = nusc_map.get_lane(closest_lane)
    closest_lane_local = get_discretized_future_lanes(ego_pos, 
                                                      ego_quat, 
                                                      lane_record, 
                                                      nusc_map, 
                                                      frame=frame, 
                                                      ego_speed=ego_speed)

    # get part of current lane that's in front of the ego
    closest_lane_local_idx = np.argmax(closest_lane_local[:,1]>0)
    closest_lane_local = closest_lane_local[closest_lane_local_idx:,:]

    # concatenate closest lane with all outgoing lanes
    ego_future_lanes = np.zeros((5, 500, 2)) # (max number of possible future lanes, points on the lanes, (x,y))
    outgoing_lane_ids = nusc_map.get_outgoing_lane_ids(closest_lane)
    for idx, lane in enumerate(outgoing_lane_ids):
        lane_record = nusc_map.get_lane(lane)
        outgoing_lane_local = get_discretized_future_lanes(ego_pos, ego_quat, lane_record, nusc_map, frame=frame, ego_speed=ego_speed)
        ego_future_lanes[idx, :closest_lane_local.shape[0], :] = closest_lane_local
        ego_future_lanes[idx, closest_lane_local.shape[0]:closest_lane_local.shape[0]+outgoing_lane_local.shape[0], :] = outgoing_lane_local

    return ego_future_lanes

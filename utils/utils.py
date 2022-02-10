from utils.utils import convert_global_coords_to_local, convert_local_coords_to_global, process_to_len
import numpy as np

def get_nearest_n_ados(obs, nbr_nearby_ados, past_steps, future_steps, init_ego_pos, init_ego_quat, ado_past_order='current_first'):
    ado_dict = {
        'ado_past': 1000. * np.ones((100, past_steps, 2)), # (nbr_ados, obs_steps, 2)
        'ado_future': 1000. * np.ones((100, future_steps, 2)), # (nbr_ados, pred_steps, 2)
        'ado_current': 1000. * np.ones((100, 2)), # (nbr_ados, 2)
        'ado_current_vel': np.zeros((100,)), #  (nbr_ados,)
        'ado_past_lane': np.zeros((100, past_steps, 2)), # (nbr_ados, obs_steps, 2)
        'ado_token': [None] * 100 # (nbr_ados, )
    }
    agent_info = obs['sensor_info']['agent_info']
    for i, ado in enumerate(agent_info):
        if len(ado['past']) == 0:
            ado['past'] = np.array([ado['translation'][:2]])
        if len(ado['future']) == 0:
            ado['future'] = np.array([ado['translation'][:2]])

        # for past and future, idx=0 is nearest to current
        ado_past_gb = np.array(ado['past'])[:,:2]
        if ado_past_order == 'distant_first':
            # TODO: check to see if this is right
            ado_past_gb = np.flip(ado_past_gb, 0).copy()
        ado_dict['ado_past'][i] = process_to_len(convert_global_coords_to_local(ado_past_gb, init_ego_pos, init_ego_quat), past_steps)
        ado_dict['ado_current'][i] = np.squeeze(convert_global_coords_to_local(np.array([ado['translation'][:2]]), init_ego_pos, init_ego_quat))
        ado_dict['ado_current_vel'][i] = ado['velocity']
        ado_dict['ado_future'][i] = process_to_len(convert_global_coords_to_local(np.array(ado['future'])[:,:2], init_ego_pos, init_ego_quat), future_steps)
        ado_dict['ado_token'][i] = ado['instance_token']

        # past lanes #
        ado_past_lane = convert_global_coords_to_local(ado['past_lane'][-past_steps:,:], init_ego_pos, init_ego_quat)

        ado_dict['ado_past_lane'][i] = ado_past_lane


    ado_current = np.array(ado_dict['ado_current']) # (nbr_ados, 2)
    # get closest ado # 
    sorted_idx = np.argsort(np.linalg.norm(ado_current, axis=-1)).flatten()
    ado_dict_array = {}
    for k, v in ado_dict.items():
        ado_dict_array[k] = np.array(v)[sorted_idx]
        if nbr_nearby_ados < ado_current.shape[0]:
            ado_dict_array[k] = ado_dict_array[k][:nbr_nearby_ados]

    return ado_dict_array

def get_nearest_intersection(obs, init_ego_pos, init_ego_quat):
    # calculate the closest intersection center #
    intersection_centers = []
    for intersection in obs['sensor_info']['map_info']['intersection']:
        a = intersection['bounding_box']
        intersection_centers.append({'traj': np.array([[(a[0]+a[2])/2, (a[1]+a[3])/2]]), 'color': 'red'})
    #### no intersection within sensing patch ####
    if len(intersection_centers) == 0:
        return np.array([1000., 1000.])

    obs['intersection_centers'] = np.array([c['traj'].flatten() for c in intersection_centers])

    intersection_local = convert_global_coords_to_local(obs['intersection_centers'], init_ego_pos, init_ego_quat)
    ego_dist = []
    for c in intersection_local:
        ego_dist.append(np.linalg.norm(c))

    idx = np.argmin(np.array(ego_dist))
    closest_intersection_center = intersection_local[idx]

    return closest_intersection_center


def get_ego_future_lanes(ego_pos_gb, ego_quat_gb, lane_pos_gb):
    lane_pos_local = convert_global_coords_to_local(lane_pos_gb, ego_pos_gb, ego_quat_gb)
    idx = np.argmax(lane_pos_local[:,1]>0)
    future_lane_local = lane_pos_local[idx:,:]
    future_lane_gb = convert_local_coords_to_global(future_lane_local, ego_pos_gb, ego_quat_gb)
    
    return future_lane_local, future_lane_gb


def get_points_on_future_lane(obs, speed, ego_pos_gb, ego_quat_gb):
    lane = obs['gt_future_lanes'][0]
    lane = lane[~np.all(lane==0, axis=1)]
    speed = int(np.floor(speed))

    lane = process_to_len(lane, 500, name='future_lane', dim=0, before_or_after='after', mode='edge')

    # TODO: what to do if the ego is stopped
    if speed < 0.01:
        speed = 1
    points_on_future_lane = get_ego_future_lanes(ego_pos_gb, ego_quat_gb, lane)[:6*speed:speed,:]
    points_on_future_lane = process_to_len(points_on_future_lane, desired_length=6, name='points_on_future_lane', dim=0)

    return points_on_future_lane

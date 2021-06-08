import numpy as np


def get_raw_data_pd_dict_from_obs(observation):
    scene_info = observation['scene_info']
    can_info = observation['sensor_info']['can_info']

    raw_data_pd_dict = {
        #### scene level ####
        'scene_token': [scene_info['scene_token']],
        'scene_description': [scene_info['scene_description']],
        'scene_name': [scene_info['scene_name']],
        'scene_nbr_samples': [scene_info['scene_nbr_samples']],
        'ego_accel_traj': [can_info['ego_accel_traj']],
        'ego_quat_traj': [can_info['ego_quat_traj']],
        'ego_pos_traj': [can_info['ego_pos_traj']],
        'ego_rotation_rate_traj': [can_info['ego_rotation_rate_traj']],
        'ego_speed_traj': [can_info['ego_speed_traj']],
        'ego_steering_deg_traj': [can_info['ego_steering_deg_traj']],
        'ego_high_level_motion': [can_info['ego_high_level_motion']],
        #### sample level ####
        'sample_time': [observation['time']],
        'sample_idx': [observation['sample_idx']],
        'sample_token': [observation['sample_token']],
        'ego_past_pos': [observation['ego_past_pos']],
        'ego_future_pos': [observation['ego_future_pos']],
        'ego_past_quat': [observation['ego_past_quat']],
        'ego_future_quat': [observation['ego_future_quat']],
        'ego_raster_img': [observation['raster_image']],
        'ego_future_lanes': [observation['gt_future_lanes']],
        'ego_road_objects': [observation['sensor_info']['ego_info']['road_objects']]
    }

    #### instance level ####
    instance_dict = {
        'instance_token': [],
        'instance_category': [],
        'instance_attribute': [],
        'instance_pos': [],
        'instance_quat': [],
        'instance_vel': [],
        'instance_past': [],
        'instance_future':[],
        'instance_road_objects':[]
    }

    nbr_ados = len(observation['sensor_info']['agent_info'])
    for agent in observation['sensor_info']['agent_info']:
        instance_dict['instance_token'].append(agent['instance_token'])
        instance_dict['instance_category'].append(agent['category'])
        instance_dict['instance_attribute'].append(agent['attribute'])
        instance_dict['instance_pos'].append(agent['translation'])
        instance_dict['instance_quat'].append(agent['rotation_quat'])
        instance_dict['instance_vel'].append(agent['velocity'])
        instance_dict['instance_past'].append(agent['past'])
        instance_dict['instance_future'].append(agent['future'])
        instance_dict['instance_road_objects'].append(agent['road_objects'])

    raw_data_pd_dict_expand = {}
    for k, v in raw_data_pd_dict.items():
        raw_data_pd_dict_expand[k] = v * nbr_ados

    raw_data_pd_dict_expand.update(instance_dict)

    return raw_data_pd_dict_expand

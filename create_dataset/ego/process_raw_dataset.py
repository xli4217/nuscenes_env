import os
import pandas as pd

import cloudpickle
import tqdm

from utils.utils import split_list_for_multi_worker, set_function_arguments, class_from_path
from utils.configuration import Configuration
from create_dataset.raw_dataset_utils import get_raw_data_pd_dict_from_obs

from collections import OrderedDict

MISALIGNED_SCENES = ['scene-0071', 'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0085', 'scene-0100', 'scene-0101', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0111', 'scene-0112', 'scene-0113', 'scene-0114', 'scene-0115', 'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264', 'scene-0276', 'scene-0302', 'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0334', 'scene-0388', 'scene-0389', 'scene-0390', 'scene-0436', 'scene-0499', 'scene-0500', 'scene-0501', 'scene-0502', 'scene-0504', 'scene-0505', 'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510', 'scene-0511', 'scene-0512', 'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518', 'scene-0547', 'scene-0548', 'scene-0549', 'scene-0550', 'scene-0551', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0730', 'scene-0731', 'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0904', 'scene-0905', 'scene-1073', 'scene-1074', 'scene-161', 'scene-162', 'scene-163', 'scene-164', 'scene-165', 'scene-166', 'scene-167', 'scene-168', 'scene-170', 'scene-171', 'scene-172', 'scene-173', 'scene-174', 'scene-175', 'scene-176', 'scene-309', 'scene-310', 'scene-311', 'scene-312', 'scene-313', 'scene-314']

NO_CANBUS_SCENES = ['scene-0'+str(s) for s in [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]]


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
        'ego_current_pos':[observation['ego_pos_gb']],
        'ego_current_quat':[observation['ego_quat_gb']],
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


def process_once(df_path_list=[], data_save_dir=None, config={}):
    env = config['other_configs']['env']
    name_or_idx = config['other_configs']['name_or_idx']
    
    info_dict_aggre = None
    for scene in tqdm.tqdm(df_path_list, 'processing scene'):
        if name_or_idx == 'idx':
            scene_name = env.nusc.scene[scene]['name']
        else:
            scene_name = scene
            
        if scene_name in NO_CANBUS_SCENES:
            continue

        if name_or_idx == 'idx':
            obs = env.reset(scene_idx=scene)
        else:
            obs = env.reset(scene_name=scene)

        done = False
        scene_info_dict_aggre = None
        while not done:
            info_dict = get_raw_data_pd_dict_from_obs(obs)

            if scene_info_dict_aggre is None:
                scene_info_dict_aggre = dict(info_dict)
            else:
                for k, v in info_dict.items():
                    scene_info_dict_aggre[k] += v
                        
            obs, done, _ = env.step()
                    
        df = pd.DataFrame(scene_info_dict_aggre)

        p = os.path.join(data_save_dir, scene_name+".pkl")
        df.to_pickle(p)
        
if __name__ == "__main__":
    pass

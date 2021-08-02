import ray
import os
import pandas as pd
import tqdm
import copy
import time

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap

NO_CANBUS_SCENES = ['scene-0'+str(s) for s in [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]]
    
def filter_instance(sample_annotation, nusc):
    category = sample_annotation['category_name']
    if 'vehicle' in  category or 'pedestrian' in category:
        attributes = [nusc.get('attribute', attribute_token)['name'] for attribute_token in sample_annotation['attribute_tokens']]
        for at in attributes:
            if 'parked' in at or 'stroller' in at or 'wheelchair' in at:
                return False
        return True
    else:
        return False

def process_once(scene_name_list=[], data_save_dir=None, config={}):
    """get raw data from dataset, everything are lists

    :param scene_name_list: list of scene names to process
    :param data_save_dir: directory to save processed raw data
    :param config: additional configs
    :returns: one pandas dataframe for each scene

    """
    nusc = ray.get(config['other_configs']['nusc'])
    helper = ray.get(config['other_configs']['helper'])
    rasterizer = ray.get(config['other_configs']['rasterizer'])
    dataroot = config['other_configs']['dataroot']
    
    nusc_can = NuScenesCanBus(dataroot=dataroot)
   
    df_dict = {
        #### scene info ####
        'scene_name': [],
        'scene_token':[],
        'scene_nbr_samples':[],
        'scene_description':[],
        'scene_log_token':[],
        'scene_map_token':[],
        'scene_vehicle_id': [],
        'scene_date_captured':[],
        'scene_location':[],
        #### sample info ####
        'sample_idx': [],
        'sample_token':[],
        'time_stamp': [],
        #### ego info ####
        'current_ego_pos': [],
        'current_ego_quat':[],
        'current_ego_accel':[],
        'current_ego_speed':[],
        'current_ego_steering': [],
        'current_ego_on_road_objects':[],
        'current_ego_raster_img':[],
        #### instance info ####
        'annotation_token': [],
        'instance_token': [],
        'instance_category': [],
        'instance_attributes': [],
        'instance_nbr_annotations':[],
        'current_instance_pos': [],
        'current_instance_quat': [],
        'current_instance_speed': [],
        'current_instance_accel':[],
        'current_instance_on_road_objects':[],
        'current_instance_raster_img':[]
    }

    #### loop over scenes ####
    for scene_name in tqdm.tqdm(scene_name_list, 'processing scene'):
        scene_df_dict = copy.deepcopy(df_dict)
        
        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
        scene = nusc.get('scene', scene_token)
        scene_log = nusc.get('log', scene['log_token'])
        scene_map = nusc.get('map', scene_log['map_token'])
        nusc_map = NuScenesMap(dataroot=dataroot, map_name=scene_log['location'])
        
        # can bus information
        if scene_name in NO_CANBUS_SCENES:
            ego_accel_traj = [None] * 50
            ego_rotation_rate_traj = [None] * 50
            ego_speed_traj = [None] * 50
        else:
            pose_msg = nusc_can.get_messages(scene_name, 'pose')
            ego_accel_traj = [pm['accel'] for pm in pose_msg][::25]
            ego_rotation_rate_traj = [pm['rotation_rate'] for pm in pose_msg][::25]
            ego_speed_traj = [pm['vel'] for pm in pose_msg][::25]
        
        #### loop over samples ####
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)

        sample_idx = 0
        t = time.time()
        while sample['next'] != '':
            # print(sample_idx)
            # print(t-time.time())
            # t = time.time()
            cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

            # ego raster
            ego_raster = rasterizer.make_input_representation(instance_token=None, sample_token=sample_token, ego=True, ego_pose=ego_pose, include_history=False)

            # ego road objects
            ego_on_road_objects = nusc_map.layers_on_point(ego_pose['translation'][0], ego_pose['translation'][1])
            
            #### loop over annotations ####
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                instance = nusc.get('instance', ann['instance_token'])

                if filter_instance(ann, nusc):
                    #### populate data ####
                    ## scene ##
                    scene_df_dict['scene_name'].append(scene_name)
                    scene_df_dict['scene_token'].append(scene_token)
                    scene_df_dict['scene_nbr_samples'].append(scene['nbr_samples'])
                    scene_df_dict['scene_description'].append(scene['description'])
                    scene_df_dict['scene_log_token'].append(scene_log['token'])
                    scene_df_dict['scene_map_token'].append(scene_map['token'])
                    scene_df_dict['scene_vehicle_id'].append(scene_log['vehicle'])
                    scene_df_dict['scene_date_captured'].append(scene_log['date_captured'])
                    scene_df_dict['scene_location'].append(scene_log['location'])
                    ## sample info ##
                    scene_df_dict['sample_idx'].append(sample_idx)
                    scene_df_dict['sample_token'].append(sample_token)
                    scene_df_dict['time_stamp'].append(sample['timestamp'])
                    ## ego info ##
                    scene_df_dict['current_ego_pos'].append(ego_pose['translation'])
                    scene_df_dict['current_ego_quat'].append(ego_pose['rotation'])
                    idx = min(sample_idx, len(ego_speed_traj)-1)
                    scene_df_dict['current_ego_speed'].append(ego_speed_traj[idx])
                    idx = min(sample_idx, len(ego_accel_traj)-1)
                    scene_df_dict['current_ego_accel'].append(ego_accel_traj[idx])
                    idx = min(sample_idx, len(ego_rotation_rate_traj)-1)
                    scene_df_dict['current_ego_steering'].append(ego_rotation_rate_traj[idx])
                    scene_df_dict['current_ego_on_road_objects'].append(ego_on_road_objects)
                    scene_df_dict['current_ego_raster_img'].append(ego_raster)
                    ## instance info ##
                    scene_df_dict['annotation_token'].append(ann_token)
                    scene_df_dict['instance_token'].append(ann['instance_token'])
                    scene_df_dict['instance_category'].append(ann['category_name'])
                    instance_attributes = [nusc.get('attribute', attribute_token)['name'] for attribute_token in ann['attribute_tokens']]
                    scene_df_dict['instance_attributes'].append(instance_attributes)
                    scene_df_dict['instance_nbr_annotations'].append(instance['nbr_annotations'])
                    scene_df_dict['current_instance_pos'].append(ann['translation'])
                    scene_df_dict['current_instance_quat'].append(ann['rotation'])
                    instance_speed = helper.get_velocity_for_agent(ann['instance_token'], sample_token)
                    scene_df_dict['current_instance_speed'].append(instance_speed)
                    instance_accel = helper.get_acceleration_for_agent(ann['instance_token'], sample_token)
                    scene_df_dict['current_instance_accel'].append(instance_accel)
                    instance_on_road_objects = nusc_map.layers_on_point(ann['translation'][0], ann['translation'][1])
                    scene_df_dict['current_instance_on_road_objects'].append(instance_on_road_objects)
                    instance_raster = rasterizer.make_input_representation(instance_token=ann['instance_token'], sample_token=sample_token, ego=False, include_history=False)
                    scene_df_dict['current_instance_raster_img'].append(instance_raster)
 
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)
            sample_idx += 1

        scene_df = pd.DataFrame(scene_df_dict)
        scene_df.to_pickle(data_save_dir+"/"+scene_name+".pkl")

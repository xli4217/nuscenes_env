import ray
import os
import numpy as np
import time
import torch

import cloudpickle
import tqdm

from utils.utils import split_list_for_multi_worker, timing_val, convert_global_coords_to_local, convert_local_coords_to_global
from utils.configuration import Configuration

import pandas as pd
from pathlib import Path

from utils.utils import assert_type, assert_shape
from nuscenes_env.nuscenes.prediction.helper import PredictHelper
from nuscenes_env.nuscenes import NuScenes
from paths import mini_path, full_path
from nuscenes_env.nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes_env.nuscenes.map_expansion import arcline_path_utils

import random


def get_data_from_sample_df(scene_name, sample_df, sample_idx, nusc, helper, more_history_data_traj):

    sample_idx = int(sample_idx)
    sample = nusc.get('sample', sample_df.iloc[0].sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    scene_log = nusc.get('log', scene['log_token'])
    nusc_map = NuScenesMap(dataroot=mini_path, map_name=scene_log['location'])

    # get ego obs info
    if np.array(sample_df.iloc[0]['ego_future_pos']).size == 0:
        return None
    ego_pos_global = np.array(sample_df.iloc[0]['ego_future_pos'][0])[:2]
    ego_quat_global = np.array(sample_df.iloc[0]['ego_future_quat'][0])
    
    # ego_vel
    idx = min(len(sample_df.iloc[0]['ego_speed_traj'])-1, int(sample_idx))
    ego_current_vel = sample_df.iloc[0]['ego_speed_traj'][idx]
    ego_future_vel = np.array(sample_df.iloc[0]['ego_speed_traj'][idx:])
    ego_past_vel = np.array(sample_df.iloc[0]['ego_speed_traj'][:idx])
    
    # ego_steering
    idx = min(len(sample_df.iloc[0]['ego_rotation_rate_traj'])-1, int(sample_idx))
    ego_current_steering = sample_df.iloc[0]['ego_rotation_rate_traj'][idx][-1]
    ego_future_steering = np.array([s[-1] for s in sample_df.iloc[0]['ego_rotation_rate_traj'][idx:]])
    ego_past_steering = np.array([s[-1] for s in sample_df.iloc[0]['ego_rotation_rate_traj'][:idx]])
    
    # ego pos and quat
    ego_vec = np.concatenate([ego_pos_global, ego_quat_global])

    # future lane information
    lane_poses_global = sample_df.iloc[0]['ego_future_lanes']
    lane_poses_local = []
    for lane in lane_poses_global:
        if lane.mean() > 1.:
            lane =  lane[~np.all(lane == 0, axis=1)] # get rid of all zero rows
            lane_local = convert_global_coords_to_local(lane, ego_pos_global, ego_quat_global)
            lane_poses_local.append(lane_local)

    # get instance obs info
    ado_dict = get_neighbor_vehicles_or_pedestrians(sample_df, nusc_map=nusc_map)
    if ado_dict is None:
        return None

    ## get ego future as target
    idx = min(len(sample_df.iloc[0]['ego_pos_traj'])-1, int(sample_idx))
    ego_future_global = np.array(sample_df.iloc[0]['ego_pos_traj'])[idx+1:, :2]
    ego_past_global = np.array(sample_df.iloc[0]['ego_pos_traj'])[:idx, :2]
    ego_future_local = convert_global_coords_to_local(ego_future_global, ego_pos_global, ego_quat_global)
    ego_past_local = convert_global_coords_to_local(ego_past_global, ego_pos_global, ego_quat_global)
    ## get ego goal
    ego_goal_global = np.array(sample_df.iloc[0]['ego_pos_traj'])[-1, :2]
    ego_goal_local = convert_global_coords_to_local(np.array([ego_goal_global]), ego_pos_global, ego_quat_global)[0]
    
    ego_road_objects = sample_df.iloc[0].ego_road_objects
    if ego_road_objects['road_segment'] != "":
        for record in nusc_map.road_segment:
            if record['token'] == ego_road_objects['road_segment'] and record['is_intersection']:
                ego_road_objects['intersection'] = record['token']

    ## res ##
    res = {
        'scene_name': scene_name,
        'sample_token': sample_df.iloc[0].sample_token,
        'sample_idx': sample_idx,
        'sample_time': sample_df.iloc[0].sample_time,
        'ego_goal': ego_goal_local,
        'ego_future': ego_future_local,
        'ego_past': ego_past_local,
        'ego_current': ego_vec,
        'ego_current_vel': ego_current_vel,
        'ego_future_vel': ego_future_vel,
        'ego_past_vel': ego_past_vel,
        'ego_current_steering': ego_current_steering,
        'ego_future_steering': ego_future_steering,
        'ego_past_steering': ego_past_steering,
        'ego_current_raster_img': sample_df.iloc[0].ego_raster_img,
        'discretized_lane': lane_poses_local,
        'ego_road_objects': ego_road_objects,
        'ego_interactions': sample_df.iloc[0].ego_interactions,
        'ado_dict': ado_dict
    }

    for k, v in more_history_data_traj.items():
        idx = min(len(v)-2, int(sample_idx))
        res['current_'+k] = v[idx]
        res['future_'+k] = np.array(v[idx+1:])
        res['past_'+k] = np.array(v[:idx])


    return res


def get_neighbor_vehicles_or_pedestrians(sample_df, nusc_map=None):
    # get instance information
    pos = []
    vel = []
    tokens = []
    ado_futures = []
    ado_past = []
    ado_road_objects = []
    ado_interactions = []
    ado_dict = {}
    
    for idx, row in sample_df.iterrows():
        ego_pos = np.array(row['ego_future_pos'][0])[:2]
        ego_quat = np.array(row['ego_future_quat'][0])

        current_ado_dict = {}
        if 'vehicle' in row['instance_category'] and 'parked' not in row['instance_attribute']:
            # get future

            #### ado map data ####
            ado_road_objects.append(row.instance_road_objects)
            if row.instance_road_objects['road_segment'] != "":
                for record in nusc_map.road_segment:
                    if record['token'] == row.instance_road_objects['road_segment'] and record['is_intersection']:
                        ado_road_objects[-1]['intersection'] = row.instance_road_objects['road_segment']
            
            current_ado_dict['road_objects'] = ado_road_objects[-1]
            
            # get future
            if row['instance_future'].shape[0] < 1:
                continue
            ado_future_global = row['instance_future']

            ado_future_local = convert_global_coords_to_local(ado_future_global, ego_pos, ego_quat)

            ado_futures.append(ado_future_local)
            current_ado_dict['future'] = ado_future_local
            
            # get past
            ado_past_global = row['instance_past']

            if ado_past_global.shape[0] < 1:
                continue
            ado_past_local = convert_global_coords_to_local(ado_past_global, ego_pos, ego_quat)
            ado_past.append(ado_past_local)
            current_ado_dict['past'] = ado_past_local
            
            if np.isnan(ado_past_global).any() or np.isnan(ado_future_global).any() or np.isnan(row.instance_pos).any() or np.isnan(row.instance_vel).any():
                continue

            # get current time step observation
            pos.append(row['instance_pos'][:2])
            vel.append(row['instance_vel'])
            tokens.append([str(row['scene_token']), str(row['sample_token']), str(row['instance_token'])])
            current_ado_dict['current_pos'] = row['instance_pos'][:2]
            current_ado_dict['current_vel'] = row['instance_vel']

            # get interactions
            ado_interactions.append(row['instance_interactions'])
            current_ado_dict['current_interactions'] = row['instance_interactions']

            ado_dict[str(row['instance_token'])] = current_ado_dict
            
    return ado_dict

def process_once(df_path_list=[], data_save_dir=None, config={}):
    nusc = config['other_configs']['nusc']
    helper = config['other_configs']['helper']

    df_list = []
    for p in df_path_list:
        df_list.append(pd.read_pickle(p))
    df = pd.concat(df_list)

    #### get training input ####
    multi_scene_df = df.set_index(['scene_name', 'sample_idx'])
    #### loop through scenes ####
    # TODO: check groupby function
    for scene_name, scene_df in tqdm.tqdm(multi_scene_df.groupby(level=0)):
        print(f"processing scene {scene_name}")
        #### loop through each step in the scene ####
        nbr_samples_in_scene = scene_df.iloc[0].scene_nbr_samples
        scene_df_dict = {}

        # gather additional history data (currently ego only)
        more_history_data_traj = {}
        for sample_idx, sample_df in scene_df.groupby(level='sample_idx'):
            if 0 < sample_idx < len(sample_df.iloc[0].ego_speed_traj):
                for k in config['other_configs']['additional_history_data']:
                    if k not in list(more_history_data_traj.keys()):
                        more_history_data_traj[k] = [sample_df.iloc[0][k]]
                    else:
                        more_history_data_traj[k].append(sample_df.iloc[0][k])
                        
        for sample_idx, sample_df in scene_df.groupby(level='sample_idx'):
            if 0 < sample_idx < len(sample_df.iloc[0].ego_speed_traj):
                out = get_data_from_sample_df(scene_name, sample_df, sample_idx, nusc, helper, more_history_data_traj) 
                if out is None:
                    continue
                    
                if len(list(scene_df_dict.keys())) == 0:
                    for k, v in out.items():
                        scene_df_dict[k] = [v]
                else:
                    for k, v in out.items():
                        scene_df_dict[k].append(v)
                    
        scene_df = pd.DataFrame(scene_df_dict)
        if data_save_dir is not None:
            scene_df.to_pickle(data_save_dir+"/" + scene_name +".pkl")
        
    return df

        
if __name__ == "__main__":
    pass

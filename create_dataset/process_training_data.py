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
from paths import mini_path
from nuscenes_env.nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes_env.nuscenes.map_expansion import arcline_path_utils

import random

nusc = NuScenes(dataroot=mini_path, version='v1.0-mini')
helper = PredictHelper(nusc)

def get_data_from_sample_df(scene_name, sample_df, sample_idx, num_closest_obs, pred_steps, obs_steps):
    sample = nusc.get('sample', sample_df.iloc[0].sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    scene_log = nusc.get('log', scene['log_token'])
    nusc_map = NuScenesMap(dataroot=mini_path, map_name=scene_log['location'])

    # get ego obs info
    ego_pos_global = np.array(sample_df.iloc[0]['ego_future_pos'][0])[:2]
    ego_quat_global = np.array(sample_df.iloc[0]['ego_future_quat'][0])
    #ego_vel = np.array([sample_df.iloc[0]['ego_speed_traj'][sample_idx]])
    ego_vel = sample_df.iloc[0]['ego_speed_traj'][sample_idx:]

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
    out = get_closest_n_moving_vehicles_or_pedestrians(sample_df, n=num_closest_obs)
    if out is None:
        return None

    if out['instance_tokens'].shape[0] <  num_closest_obs:
        instance_tokens = np.vstack([out['instance_tokens'], -1*np.ones((num_closest_obs - out['instance_tokens'].shape[0], 3))])
        out['instance_tokens'] = instance_tokens

    if np.isnan(out['ado_obs']).any():
        return None

    ## get ego future as target
    ego_future_global = np.array(sample_df.iloc[0]['ego_pos_traj'])[sample_idx+1:sample_idx+pred_steps+1, :2]
    ego_past_global = np.array(sample_df.iloc[0]['ego_pos_traj'])[sample_idx-obs_steps:sample_idx, :2]
    ego_future_local = convert_global_coords_to_local(ego_future_global, ego_pos_global, ego_quat_global)
    ego_past_local = convert_global_coords_to_local(ego_past_global, ego_pos_global, ego_quat_global)


    ## res ##
    res = {
        'scene_name': scene_name,
        'sample_token': sample_df.iloc[0].sample_token,
        'sample_idx': sample_idx,
        'sample_time': sample_df.iloc[0].sample_time,
        'ego_future': ego_future_local,
        'ego_past': ego_past_local,
        'ego_observations': ego_vec,
        'ego_vel': ego_vel,
        'discretized_lane': lane_poses_local
    }
    res.update(out)

    return res


def get_closest_n_moving_vehicles_or_pedestrians(sample_df, n=5):

    # get instance information
    pos = []
    vel = []
    tokens = []
    ado_futures = []
    ado_past = []
    for idx, row in sample_df.iterrows():
        ego_pos = np.array(row['ego_future_pos'][0])[:2]
        ego_quat = np.array(row['ego_future_quat'][0])

        if 'vehicle' in row['instance_category'] and 'parked' not in row['instance_attribute']:
            # get future
            ado_future_global = row['instance_future']

            if ado_future_global.shape[0] != 12:
                continue
            ado_future_local = convert_global_coords_to_local(ado_future_global, ego_pos, ego_quat)

            ado_futures.append(ado_future_local)

            # get past
            ado_past_global = row['instance_past']

            if ado_past_global.shape[0] != 4:
                continue
            ado_past_local = convert_global_coords_to_local(ado_past_global, ego_pos, ego_quat)
            ado_past.append(ado_past_local)

            # get current time step observation
            pos.append(row['instance_pos'][:2])
            vel.append(row['instance_vel'])
            tokens.append([str(row['scene_token']), str(row['sample_token']), str(row['instance_token'])])


    inst_pos = np.array(pos)
    inst_vel = np.array(vel)
    ado_futures = np.array(ado_futures)
    ado_past = np.array(ado_past)
    tokens = np.array(tokens)

    if len(inst_pos) == 0:
        return None
    dist = np.linalg.norm(inst_pos - ego_pos, axis=1)
    idx = np.argsort(dist)

    n = min(n, len(idx))
    closest_n_ado_futures = ado_futures[idx, :][:n,:]
    closest_n_ado_past = ado_past[idx, :][:n,:]
    closest_n_inst_pos = inst_pos[idx, :][:n, :]
    closest_n_inst_vel = inst_vel[idx][:, np.newaxis][:n, :]
    closest_n_tokens = tokens[idx, :][:n,:]

    closest_n_inst_pos_local = convert_global_coords_to_local(closest_n_inst_pos,ego_pos, ego_quat) 

    inst_obs = np.hstack([closest_n_inst_pos_local, closest_n_inst_vel])

    if len(idx) < n:
        inst_obs = np.vstack([inst_obs, np.zeros((n-len(idx), inst_obs.shape[1]))])
        # closest_n_tokens = np.vstack([closest_n_tokens, np.zeros((n-len(idx), 3))])

    out = {
        'ado_obs': inst_obs,
        'ado_futures': closest_n_ado_futures,
        'ado_past': closest_n_ado_past,
        'ado_vel': closest_n_inst_vel,
        'instance_tokens': closest_n_tokens
    }

    return out


class ProcessTrainingData(object):

    def __init__(self, config={}):
        #### common setup ####
        self.config = {
            'obs_steps':4,
            'pred_steps': 6,
            'freq':2,
            'num_closest_obs': 4,
            'raw_data_path': None,
            'data_save_dir': None,
        }

        self.config.update(config)

        if self.config['raw_data_path'] is None:
            raise ValueError('raw data path not provided')

        self.raw_data_df = pd.read_pickle(self.config['raw_data_path'])


    def process(self):
        #### Trainig Input Initialize ####
        df_dict = {}
        #### get training input ####
        multi_scene_df = self.raw_data_df.set_index(['scene_name', 'sample_idx'])
        #### loop through scenes ####
        for scene_name, scene_df in tqdm.tqdm(multi_scene_df.groupby(level=0)):
            print(f"processing scene {scene_name}")
            #### loop through each step in the scene ####
            for sample_idx, sample_df in scene_df.groupby(level=1):
                if self.config['obs_steps'] < sample_idx < sample_df.iloc[0].scene_nbr_samples - self.config['pred_steps'] - 1:
                    # sample_idx data is current data, sample_idx+1:sample_idx+pred_steps+1 is future data

                    out = get_data_from_sample_df(scene_name, sample_df, sample_idx, self.config['num_closest_obs'], self.config['pred_steps'], self.config['obs_steps']) 
                    if out is None:
                        continue

                    if len(list(df_dict.keys())) == 0:
                        for k, v in out.items():
                            df_dict[k] = [v]
                    else:
                        for k, v in out.items():
                            df_dict[k].append(v)

                else:
                    continue


        df = pd.DataFrame(df_dict)
        df.to_pickle(self.config['data_save_dir'])

if __name__ == "__main__":
    import os

    config = {
        'obs_steps':2,
        'pred_steps': 6,
        'freq':2,
        'num_closest_obs': 4,
        'raw_data_path': os.path.join(os.environ['PKG_PATH'], 'create_dataset', 'raw_dataset', 'raw_dataset.pkl'),
        'data_save_dir': os.path.join(os.environ['PKG_PATH'], 'create_dataset', 'processed_dataset', 'processed_dataset.pkl'),
    }

    cls = ProcessTrainingData(config)
    cls.process()




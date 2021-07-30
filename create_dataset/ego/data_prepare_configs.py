import os
from pathlib import Path
import fire
import pandas as pd
from create_dataset.ray_data_processor import RayDataProcessor

from create_dataset.vehicle_behavior_filter.filters.scenario_filters import *
from create_dataset.vehicle_behavior_filter.filters.interaction_filters import *
from create_dataset.vehicle_behavior_filter.filters.maneuver_filters import *

def get_config(dataset_type='mini',
               data_root_dir=os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df'),
               ego_or_ado='ego',
               num_workers=30,
               mode='raw'
):
    

    NUM_WORKERS = num_workers

    if dataset_type == 'mini':
        version = 'v1.0-mini'
    elif dataset_type == 'full':
        version = 'v1.0'

    data_root_dir = os.path.join(data_root_dir, 'ego_or_ado')
        
    dir_raw = os.path.join(data_root_dir, 'raw', 'scene_df')
    if not os.path.isdir(dir_raw):
        os.makedirs(dir_raw)

    dir_filter = os.path.join(data_root_dir, 'filtered', 'scene_df')
    if not os.path.isdir(dir_filter):
        os.makedirs(dir_filter)

    dir_processed = os.path.join(data_root_dir, 'processed', 'scene_df')
    if not os.path.isdir(dir_processed):
        os.makedirs(dir_processed)

    dir_final = os.path.join(data_root_dir, 'final', 'scene_df')
    if not os.path.isdir(dir_final):
        os.makedirs(dir_final)


    if mode == 'raw':
        #### ProcessRawData config ####
        config = {
            'env_config': env_config,
            'input_data_dir': None,
            'output_data_dir': dir_raw,
            'num_workers': NUM_WORKERS,
            'other_configs':{},
            'process_once_func': 'create_dataset.process_raw_dataset.process_once'
        }

    elif mode == 'filter':
        # #### Filter data config ####    
        config = {
            'input_data_dir': dir_raw,
            'output_data_dir': dir_filter,
            'num_workers': NUM_WORKERS,
            'other_configs':{
                'scenario_filters': {'in_intersection': is_in_intersection},
                'interaction_filters': {'follows': is_follow, 'yields': is_yielding},
                'maneuver_filters': {'turn_right': is_turning_right, 'turn_left': is_turning_left},
            },
            'process_once_func': 'create_dataset.vehicle_behavior_filter.filters.filter_api.run_once'
        }

    elif mode == 'process':
        #### Processed Data config ####
        config = {
            'version':version,
            'load_data': True,
            'input_data_dir': dir_filter,
            'output_data_dir': dir_processed,
            'num_workers': NUM_WORKERS,
            'other_configs':{
                'additional_history_data': ['ego_raster_img', 'ego_maneuvers']
            },
            'process_once_func': 'create_dataset.process_training_data.process_once'
        }

    elif mode == 'final':
        #### Process Final  Data config ####
        config = {
            'input_data_dir': dir_processed,
            'output_data_dir': dir_final,
            'num_workers': NUM_WORKERS,
            'other_configs':{
                'nb_closest_ado': 6,
                'obs_steps': 4,
                'pred_steps': 6
            },
            'process_once_func': 'create_dataset.process_final_data.process_once'
        }


    else:
        raise ValueError()

    return config

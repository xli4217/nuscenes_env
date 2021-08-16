import os
from pathlib import Path
import fire
import pandas as pd
from create_dataset.ray_data_processor import RayDataProcessor
from paths import mini_path, full_path

from create_dataset.filters.scenario_filters import *
from create_dataset.filters.interaction_filters import *
from create_dataset.filters.maneuver_filters import *

def set_dir(data_root_dir=None, test=False):
    if data_root_dir is None:
        data_root_dir = os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df')

    if test:
        test_dir = 'test'
    else:
        test_dir = ''
        
    dir_raw = os.path.join(data_root_dir, 'raw', 'scene_df', test_dir, 'pkl')
    if not os.path.isdir(dir_raw):
        os.makedirs(dir_raw)

    dir_filter = os.path.join(data_root_dir, 'filtered', 'scene_df', test_dir)
    if not os.path.isdir(dir_filter):
        os.makedirs(dir_filter)

    dir_training = os.path.join(data_root_dir, 'training', 'scene_df', test_dir)
    if not os.path.isdir(dir_training):
        os.makedirs(dir_training)

    dir_final = os.path.join(data_root_dir, 'final', test_dir)
    if not os.path.isdir(dir_final):
        os.makedirs(dir_final)

    return dir_raw, dir_filter, dir_training, dir_final

def get_config(dataset_type='full',
               data_root_dir=None,
               num_workers=30,
               mode='raw',
               test=False
):
    
    dir_raw, dir_filter, dir_training, dir_final = set_dir(data_root_dir, test)
    
    NUM_WORKERS = num_workers

    if dataset_type == 'mini':
        version = 'v1.0-mini'
    elif dataset_type == 'full':
        version = 'v1.0'



    #### NuScenesAgent config ####
    na_config = RawData_NuScenesAgent_config = {
        'version':version,
        'debug': False,
        'pred_horizon': 6,
        'obs_horizon': 2,
        'freq': 2,
        'load_dataset': True,
        'debug': False,
        'py_logging_path': None,
        'tb_logging_path': None
    }


    #### NuScenesEnv config ####
    env_config = {
        'NuScenesAgent_config':RawData_NuScenesAgent_config,
        'Sensor_config': {'sensing_patch_size': (60,60), 'agent_road_objects': True},
        'render_paper_ready': False,
        'render_type': [],
        #'render_type': [],
        #'render_elements': ['sensor_info'],
        'patch_margin': 35,
        'save_image_dir': os.path.join(os.environ['PKG_PATH'], 'dataset', 'raw', 'image'),
        'all_info_fields': ['center_lane', 'raster_image'],
        #'all_info_fields': ['center_lane']
    }

    if mode == 'raw':
        #### ProcessRawData config ####
        config = {
            'version': version,
            'load_data': True,
            'input_data_dir': None,
            'output_data_dir': dir_raw,
            'num_workers': NUM_WORKERS,
            'process_from_scratch': False,
            'other_configs':{
                'dataroot': full_path
            },
            'process_once_func': 'create_dataset.process_raw.process_once'
        }

    elif mode == 'filter':
        # #### Filter data config ####    
        config = {
            'input_data_dir': dir_raw,
            'output_data_dir': dir_final,
            'num_workers': NUM_WORKERS,
            'other_configs':{
                'categories': ['vehicle'],
                'attributes': ['stopped', 'moving'],
                'scenarios': ['intersection'],
                'scenario_filter': 'create_dataset.filters.scenario_filters.scenario_filter',
                'interaction_filter_range': 30,
                'maneuver_filters': {
                    'ego_maneuver_filter':ego_maneuver_filter,
                    'ado_maneuver_filter': ado_maneuver_filter
                },
                'obs_steps': 4,
                'pred_steps': 6,
                'nb_closest_neighbors': 6,
                'max_neighbor_range': 40
            },
            'process_once_func': 'create_dataset.process_filtered.process_once'
        }

    elif mode == 'training':
        #### Process Final  Data config ####
        config = {
            'input_data_dir': dir_filter,
            'output_data_dir': dir_training,
            'num_workers': NUM_WORKERS,
            'other_configs':{
                'obs_steps': 4,
                'pred_steps': 6
            },
            'process_once_func': 'create_dataset.process_type_and_shape.process_once'
        }

    elif mode == 'final':
        config = {
            'training_data_dir': dir_training,
            'save_dir': dir_final,
            'train_val_split_filter':{
                'type': 'create_dataset.dataset_utils.train_val_split_filter',
                'config': {}
            }, 
            'additional_processor': {
                'type': 'create_dataset.dataset_utils.data_processor',
                'config': {
                    'nb_closest_neighbors': 6,
                    'max_neighbor_range': 40
                }
            },
            'normalize_elements': {
                # 'past_ego_speed':1,
                # 'current_ego_speed': 1,
                # 'past_ego_steering': 2,
                # 'current_ego_steering': 2
            }
        }

        
    else:
        raise ValueError()

    return config

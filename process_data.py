import os
from configs.configs import *
import fire

def create(mode='raw'):
    dataset_type = 'mini'
    #dataset_type = 'full'

    if dataset_type == 'mini':
        version = 'v1.0-mini'
    elif dataset_type == 'full':
        version = 'v1.0'

    #scene_name = '_scene-0900'
    scene_name = ""

    dir_raw = os.path.join(os.environ['PKG_PATH'], 'dataset', 'raw', 'scene_df')
    dir_filter = os.path.join(os.environ['PKG_PATH'], 'dataset', 'filtered', 'scene_df')

    if not os.path.isdir(dir_raw):
        os.makedirs(dir_raw)
    if not os.path.isdir(dir_filter):
        os.makedirs(dir_filter)


    #### NuScenesAgent config ####
    RawData_NuScenesAgent_config = {
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
        #'all_info_fields': ['center_lane', 'raster_image'],
        'all_info_fields': ['center_lane']
    }

    #### ProcessRawData config ####
    process_raw_data_config = {
        'Env':{
            'type': "external_libs.nuscenes_env.env.env.NuScenesEnv",
            'config': env_config
        },
        'num_workers': 20,
        'data_save_dir':dir_raw,
        'get_raw_data_pd_dict_from_obs': "create_dataset.raw_dataset_utils.get_raw_data_pd_dict_from_obs"
    }

    # #### Filter data config ####
    from create_dataset.vehicle_behavior_filter.filters.scenario_filters import is_in_intersection
    from create_dataset.vehicle_behavior_filter.filters.interaction_filters import is_follow, is_yielding

    filter_data_config = {
        'raw_data_dir': dir_raw,
        'filtered_data_save_dir': dir_filter,
        'scenario_filters': {'in_intersection': is_in_intersection},
        'interaction_filters': {'follows': is_follow, 'yields': is_yielding},
        'num_workers': 10
    }

    if mode == 'raw':
        from create_dataset.process_raw_dataset import ProcessRawDataset

        cls = ProcessRawDataset(config=process_raw_data_config)
        cls.get_raw_data()
    elif mode == 'filter':
        from create_dataset.vehicle_behavior_filter.filters.filter_api import FilterApi
        from configs.configs import filter_data_config

        filter = FilterApi(filter_data_config)
        filter.run()


if __name__ == "__main__":
    fire.Fire(create)

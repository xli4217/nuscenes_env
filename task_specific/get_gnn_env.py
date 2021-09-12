from env.dataset_env import NuScenesDatasetEnv
import os
from pathlib import Path

def get_env():
    dir_raw = os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df', 'raw', 'scene_df')
    dir_final = os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df', 'final')
    dir_raster = dir_raw + '/png'

    #### NuScenesAgent config ####
    na_config = RawData_NuScenesAgent_config = {
        'version':'v1.0',
        'debug': False,
        'pred_horizon': 6,
        'obs_horizon': 2,
        'freq': 2,
        'load_dataset': True,
        'debug': False,
        'py_logging_path': None,
        'tb_logging_path': None
    }


    dataset_env_config = {
        'NuScenesAgent_config':RawData_NuScenesAgent_config,
        'data_dir': dir_final,
        'raster_dir': dir_raster,
        'render_paper_ready': True,
        'render_type': ['image'],
        'render_elements': ['sim_ego', 'interaction_labels', 'token_labels', 'groundtruth'],
        'patch_margin': 35,
        'save_image_dir': None,
        'all_info_fields': ['raster_image'],
        'control_mode': 'position',
        'train_dataset_path': os.path.join(os.environ['PKG_PATH'], 'agn2','dataset', 'icra', 'nuscenes_split', 'gnn_train.pkl'),
        'val_dataset_path': os.path.join(os.environ['PKG_PATH'], 'agn2','dataset', 'icra', 'nuscenes_split', 'gnn_val.pkl')}

    env = NuScenesDatasetEnv(dataset_env_config)

    return env

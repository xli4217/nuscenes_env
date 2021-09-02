import os
from env.dataset_env2 import DatasetEnv2
from pathlib import Path

train_p = os.path.join(os.environ['PKG_PATH'], os.environ['PKG_NAME'], 'dataset', os.environ['PKG_NAME'], 'gnn_train.pkl')
val_p = os.path.join(os.environ['PKG_PATH'], os.environ['PKG_NAME'], 'dataset', os.environ['PKG_NAME'], 'gnn_val.pkl')


na_config = {
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


dir_raw = os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df', 'raw', 'scene_df')
dir_raster = dir_raw + '/png'

config = {
    'NuScenesAgent_config': na_config,
    'train_dataset_path': train_p,
    'val_dataset_path': val_p,
    'raster_dir': dir_raster
}

env = DatasetEnv2(config)
env.reset(scene_name='scene-0065')

env.step()

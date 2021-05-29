import os
import numpy as np
import time
import torch
import plotly.express as px
import pandas as pd

import cloudpickle
import tqdm
import matplotlib.pyplot as plt

from utils.utils import split_list_for_multi_worker, timing_val, set_function_arguments, class_from_path
from utils.configuration import Configuration
from create_dataset.dataset_utils import get_raw_data_pd_dict_from_obs

from collections import OrderedDict

import pandas as pd
import pathlib
import ray
from paths import mini_path, full_path

MISALIGNED_SCENES = ['scene-0071', 'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0085', 'scene-0100', 'scene-0101', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0111', 'scene-0112', 'scene-0113', 'scene-0114', 'scene-0115', 'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264', 'scene-0276', 'scene-0302', 'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0334', 'scene-0388', 'scene-0389', 'scene-0390', 'scene-0436', 'scene-0499', 'scene-0500', 'scene-0501', 'scene-0502', 'scene-0504', 'scene-0505', 'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510', 'scene-0511', 'scene-0512', 'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518', 'scene-0547', 'scene-0548', 'scene-0549', 'scene-0550', 'scene-0551', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0730', 'scene-0731', 'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0904', 'scene-0905', 'scene-1073', 'scene-1074', 'scene-161', 'scene-162', 'scene-163', 'scene-164', 'scene-165', 'scene-166', 'scene-167', 'scene-168', 'scene-170', 'scene-171', 'scene-172', 'scene-173', 'scene-174', 'scene-175', 'scene-176', 'scene-309', 'scene-310', 'scene-311', 'scene-312', 'scene-313', 'scene-314']

NO_CANBUS_SCENES = ['scene-0'+str(s) for s in [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]]


class ProcessRawDataset(object):

    def __init__(self, config={}):
        self.config = Configuration({
            'Env':{
                'type': None,
                'config': {}
            },
            'num_workers': 1,
            'data_save_dir': ""
        })

        self.config.update(config)

        self.env = class_from_path(self.config['Env']['type'])(config=self.config['Env']['config'])
        self.nb_scenes = len(self.env.nusc.scene)

        if self.config['num_workers'] > 1:
            ray.shutdown()
            if os.environ['COMPUTE_LOCATION'] == 'local':
                ray.init()
            else:
                ray.init(temp_dir=os.path.join(os.environ['HOME'], 'ray_tmp'), redis_max_memory=10**9, object_store_memory=100*10**9)

            self.env = ray.put(self.env)

    def get_raw_data(self, data_save_dir:str=None, scene_name=None):
        data_save_dir, = set_function_arguments(OrderedDict(data_save_dir=data_save_dir), self.config.config)

        @ray.remote
        def get_raw_data_worker(env, name_or_idx='idx', worker_list:list=[]):
            info_dict_aggre = None
            for scene in tqdm.tqdm(worker_list, 'processing scene'):
                print(f"scene:{scene}")
                if scene in NO_CANBUS_SCENES:
                    continue
                if name_or_idx == 'idx':
                    obs = env.reset(scene_idx=scene)
                else:
                    obs = env.reset(scene_name=scene)
                done = False
                while not done:
                    info_dict = get_raw_data_pd_dict_from_obs(obs)
                    if info_dict_aggre is None:
                        info_dict_aggre = dict(info_dict)
                    else:
                        for k, v in info_dict.items():
                            info_dict_aggre[k] += v
                    obs, done = env.step()

            return info_dict_aggre

        if scene_name is not None:
            worker_lists = [[scene_name]]
            obj_refs = [get_raw_data_worker.remote(self.env, 'name', worker_list) for worker_list in worker_lists]
        else:
            worker_lists = split_list_for_multi_worker(list(range(self.nb_scenes)), self.config['num_workers'])
            obj_refs = [get_raw_data_worker.remote(self.env, 'idx', worker_list) for worker_list in worker_lists]

        ready_refs, remaining_refs = ray.wait(obj_refs, num_returns=len(worker_lists), timeout=None)
        aggregated_info_dict = None
        for ready_ref in ready_refs:
            info_dict = ray.get(ready_ref)

            if aggregated_info_dict is None:
                aggregated_info_dict = dict(info_dict)
            else:
                for k, v in aggregated_info_dict.items():
                    aggregated_info_dict[k] += info_dict[k]

        df = pd.DataFrame(aggregated_info_dict)
        print(f"Dataframe size: {df.shape}")
        df.to_pickle(data_save_dir+"/raw_dataset.pkl")


if __name__ == "__main__":
    from nuscenes_env.env.env import NuScenesEnv
    import os

    RawData_NuScenesAgent_config = {
        'debug': False,
        'pred_horizon': 6,
        'obs_horizon': 2,
        'freq': 2,
        'load_dataset': True,
        'debug': False,
        'py_logging_path': None,
        'tb_logging_path': None
    }


    config = {
        'Env':{
            'type': "external_libs.nuscenes_env.env.env.NuScenesEnv",
            'config': {
                'NuScenesAgent_config':RawData_NuScenesAgent_config,
                'Sensor_config':{},
                'SceneGraphics_config':{},
                #'all_info_fields': ['center_lane', 'raster_image']
                'all_info_fields': ['center_lane']
            }
        },
        'num_workers': 2,
        'data_save_dir': os.path.join(os.environ['PKG_PATH'], 'create_dataset', 'raw_dataset')
    }

    cls = ProcessRawDataset(config=config)
    cls.get_raw_data(scene_name='scene-0061')

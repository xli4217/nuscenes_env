import ray
import numpy as np
import os
import pandas as pd
from utils.utils import split_list_for_multi_worker, timing_val, set_function_arguments, class_from_path

from pathlib import Path
from paths import mini_path, full_path

from nuscenes_env.nuscenes.prediction.helper import PredictHelper
from nuscenes_env.nuscenes import NuScenes
from input_representation.static_layers import StaticLayerRasterizer
from input_representation.agents import AgentBoxesWithFadedHistory
from input_representation.interface import InputRepresentation
from input_representation.combinators import Rasterizer


class RayDataProcessor(object):

    def __init__(self, config={}):
        self.config = {
            'version': 'v1.0-mini',
            'load_data': False,
            'input_data_dir': None,
            'output_data_dir': None,
            'num_workers': 1,
            'scene_name': None, # process only this scene - for testing 
            'process_from_scratch': True,
            'other_configs': {
            },
            'process_once_func': ""
        }

        self.config.update(config)
        self.nusc = None
        self.helper = None
        self.rasterizer = None
        #### load nuscenes ####
        if self.config['load_data']:
            if self.config['version'] == 'v1.0-mini':
                self.nusc = NuScenes(dataroot=mini_path, version='v1.0-mini')
            if self.config['version'] == 'v1.0':
                self.nusc = NuScenes(dataroot=full_path, version='v1.0')
            self.helper = PredictHelper(self.nusc)
            static_layer_rasterizer = StaticLayerRasterizer(self.helper, resolution=0.2)
            agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=2, resolution=0.2)
            self.rasterizer = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        # get total worker list #
        if self.config['input_data_dir'] is not None:
            self.input_data_fn = [str(p) for p in Path(self.config['input_data_dir']).rglob('*.pkl')]
        else:
            # currently on used in process_raw
            self.input_data_fn = [scene['name'] for scene in self.nusc.scene]
            if not self.config['process_from_scratch']:
                processed_scenes = [str(p).split('/')[-1][:-4] for p in Path(self.config['output_data_dir']).rglob('*.pkl')]
                actual_input_data_fn = []
                for scene_name in self.input_data_fn:
                    if scene_name not in processed_scenes:
                        actual_input_data_fn.append(scene_name)
                self.input_data_fn = actual_input_data_fn
                
        #### configure Ray ####
        if self.config['num_workers'] > 1:
            ray.shutdown()
            if os.environ['COMPUTE_LOCATION'] == 'local':
                ray.init()
            else:
                #ray.init(temp_dir=os.path.join(os.environ['HOME'], 'ray_tmp'), redis_max_memory=10**9, object_store_memory=100*10**9)
                ray.init(temp_dir=os.path.join(os.environ['HOME'], 'ray_tmp'))
                
            self.process_once_func = ray.remote(class_from_path(self.config['process_once_func']))
            #### initialize nusc ####
            if self.nusc is not None:
                self.nusc = ray.put(self.nusc)
            if self.helper is not None:
                self.helper = ray.put(self.helper)
            if self.rasterizer is not None:
                self.rasterizer = ray.put(self.rasterizer)
        else:
            self.process_once_func = class_from_path(self.config['process_once_func'])

        self.config['other_configs']['nusc'] = self.nusc
        self.config['other_configs']['helper'] = self.helper
        self.config['other_configs']['rasterizer'] = self.rasterizer
            
    def run(self):
        if self.config['num_workers'] > 1:
            worker_lists = split_list_for_multi_worker(self.input_data_fn, self.config['num_workers'])
        
            obj_refs = [self.process_once_func.remote(worker_list, self.config['output_data_dir'], self.config) for worker_list in worker_lists]

            ready_refs, remaining_refs = ray.wait(obj_refs, num_returns=len(worker_lists), timeout=None)
        else:
            self.process_once_func(self.input_data_fn, self.config['output_data_dir'], self.config)


    

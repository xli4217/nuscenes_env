import os
import json


from external_libs.nuscenes.nuscenes import NuScenes
from external_libs.nuscenes.prediction import PredictHelper
from external_libs.nuscenes.map_expansion.map_api import NuScenesMap
from external_libs.nuscenes.can_bus.can_bus_api import NuScenesCanBus

#import logging
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from paths import nuscenes_ds_paths, mini_path, full_path
from utils.configuration import Configuration

class NuScenesAgent(object):

    def __init__(self, config:dict={}, helper:PredictHelper=None, py_logger=None, tb_logger=None):
        self.na_config = Configuration({
            'debug': False,
            'pred_horizon': 6,
            'obs_horizon': 2,
            'freq': 2,
            'load_dataset': False,
            'version': 'v1.0-mini',
            'debug': False,
            'py_logging_path': None,
            'tb_logging_path': None
        })

        self.na_config.update(config)
        self.name = None

        self.py_logger = py_logger
        self.tb_logger=tb_logger

        self.dataroot = None
        if 'mini' in self.na_config['version']:
            self.dataroot = mini_path
        else:
            self.dataroot = full_path
        if self.py_logger is None and self.na_config['py_logging_path'] is not None:
            print(f"py logging path: {self.na_config['py_logging_path']}")
            self.py_logger = logger
            self.py_logger.add(self.na_config['py_logging_path']+"/log.txt")
            
        #     self.py_logger = logging.getLogger(self.name)
        #     print(f"py logging path: {self.na_config['py_logging_path']}")
        #     self.py_logger.addHandler(logging.FileHandler(os.path.join(self.na_config['py_logging_path'], 'log.txt'),  mode='w'))
        # if self.py_logger is not None:
        #     self.py_logger.propagate = False
            
        if self.tb_logger is None and self.na_config['tb_logging_path'] is not None:
            self.tb_logger = SummaryWriter(log_dir=os.path.join(self.na_config['tb_logging_path']))

        self.helper = helper
        self.nusc = None    
        if self.helper is not None:
            self.nusc = self.helper.data
        else:
            if self.dataroot is not None and self.na_config['version'] is not None and self.na_config['load_dataset'] and self.helper is None:
                self.nusc = NuScenes(dataroot=self.dataroot, version=self.na_config['version'], verbose=True)
                self.helper = PredictHelper(self.nusc)

        #### Initialize Map ####
        self.nusc_map_dict = {
            'boston-seaport': NuScenesMap(dataroot=self.dataroot, map_name='boston-seaport'),
            'singapore-hollandvillage':NuScenesMap(dataroot=self.dataroot, map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=self.dataroot, map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=self.dataroot, map_name='singapore-queenstown'),
        }

        #### Initialize CAN ####
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)

        ####
        self.all_info = {
            'config': self.na_config
        }
        
        # self.update_all_info()
        
    def update_all_info(self):
        raise NotImplementedError("")

    def save_config(self, config_save_path:str=None):
        if config_save_path is not None:
            json.dump(self.na_config, open(config_save_path, 'w'))

                    

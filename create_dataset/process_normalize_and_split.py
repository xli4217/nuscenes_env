import ray
import os
import numpy as np
import pandas as pd
import cloudpickle
from PIL import Image

import matplotlib.pyplot as plt
from pathlib import Path
from utils.utils import get_dataframe_summary, process_to_len, class_from_path, split_list_for_multi_worker
import tqdm

class ProcessDatasetSplit(object):
    def __init__(self, config={}, py_logger=None):
        self.config = {
            'input_data_dir': "",
            'save_dir': "",
            'num_workers': 1,
            # key is column name in the dataframe, value is normalized range (to [0, value])
            # 'normalize_elements': {'current_ego_speed': 1, 'current_ego_steering': 2},
            'normalize_elements': {},
            'train_val_split_filter': {
                'type': None,
                'config': {}
            },
            'additional_processor': {
                'type': None,
                'config': {}
            }
        }
        self.config.update(config)
        self.py_logger = py_logger
        if py_logger is not None:
            self.py_logger.info(f"ProcessDatasetSplit config: {self.config}")
    
        if self.config['train_val_split_filter']['type'] is None:
            raise ValueError('train val split filter not provided')
        self.train_val_split_filter = class_from_path(self.config['train_val_split_filter']['type'])
        
        self.additional_processor = None
        if self.config['additional_processor']['type'] is not None:
            self.additional_processor = class_from_path(self.config['additional_processor']['type'])
    

        self.final_data_fn = [str(p) for p in Path(self.config['input_data_dir']).rglob('*.pkl')]

        #### initialize ray #####
        if self.config['num_workers'] > 1:
            ray.shutdown()
            if os.environ['COMPUTE_LOCATION'] == 'local':
                ray.init()
            else:
                if os.environ['COMPUTE_LOCATION'] == 'satori':
                    ray.init(temp_dir=os.path.join(os.environ['HOME'], 'ray_tmp'))
                else:
                    ray.init(_temp_dir=os.path.join(os.environ['HOME'], 'ray_tmp'), include_dashboard=False)
                    
            self.additional_processor = ray.remote(self.additional_processor)


    def normalize(self, df, normalize_elements={'ego_current_vel': 1, 'ego_current_steering': 2}):
        print(f"normalize_elements: {normalize_elements}")
        element_min_max = {}
        for k, v in normalize_elements.items():
            if 'img' not in k:
                # df['origin_'+k] = df[k]
                element = np.array(df[k].tolist())
                element = element.reshape(-1, element.shape[-1])
                
                element_min_max[k] = {
                    'min': np.min(element, axis=0),
                    'max': np.max(element, axis=0),
                    'mean': np.mean(element, axis=0),
                    'std': np.std(element, axis=0),
                    'scale':v
                }

                #df = df.apply(lambda x: v*(x-element_min)/(element_max-element_min) if x.name == k else x)
                # df = df.apply(lambda x: v*(x-element_mean)/element_std if x.name == k else x)

        # save image mean and variance for normalization #
        raster_paths = [str(p) for p in df.current_agent_raster_path.tolist()]
        raster = np.array([np.asarray(Image.open(os.path.join(self.config['raster_dir'], p))) for p in raster_paths])
        raster = np.transpose(raster, (0, 3, 1, 2))
        data_size, channels, w, h = raster.shape
        raster = raster.reshape(data_size, channels, w*h)
        raster = np.transpose(raster, (1,2,0))
        raster = raster.reshape(channels, w*h*data_size)
        raster_mean = raster.mean(axis=-1)
        raster_std = raster.std(axis=-1)
        element_min_max['raster'] = {'mean': raster_mean, 'std': raster_std}
        
        cloudpickle.dump(element_min_max, open(self.config['save_dir']+"/min_max.pkl", 'wb'))
        return df
    
    def process(self):
        """turn processed dataset into final dataframe usable by the model

        :returns: a pd dataframe 
        """

        if self.additional_processor is None:
            raise ValueError('please provide a data processor')

        if self.config['num_workers'] == 1:

            df = self.additional_processor(self.final_data_fn, self.config['additional_processor']['config'])
        else:
            worker_lists = split_list_for_multi_worker(self.final_data_fn, self.config['num_workers'])
            obj_refs = [self.additional_processor.remote(worker_list, self.config) for worker_list in worker_lists]

            ready_refs, remaining_refs = ray.wait(obj_refs, num_returns=len(worker_lists), timeout=None)

            df_aggre = []
            for ref in ready_refs:
                df_i = ray.get(ref)
                df_aggre.append(df_i)

            df = pd.concat(df_aggre)
            
        df.reset_index(drop=True, inplace=True)
        
        train_df, val_df = self.create_train_val_split(df)
        print(f"train_df shape is {train_df.shape}")
        print(f"val_df shape is {val_df.shape}")

        mini_train_df = train_df.iloc[:300,:]
        mini_val_df = val_df.iloc[:70,:]
        
        train_df.to_pickle(self.config['save_dir']+"/train.pkl")
        val_df.to_pickle(self.config['save_dir']+"/val.pkl")

        mini_train_df.to_pickle(self.config['save_dir']+"/mini_train.pkl")
        mini_val_df.to_pickle(self.config['save_dir']+"/mini_val.pkl")

        
    def create_train_val_split(self, df):
        normalized_df = self.normalize(df, normalize_elements=self.config['normalize_elements'])
        train_df, val_df = self.train_val_split_filter(normalized_df, self.config['train_val_split_filter']['config'])

        normalized_df.to_pickle(self.config['save_dir']+"/full.pkl")

        return train_df, val_df

        

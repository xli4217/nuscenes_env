import numpy as np
from numpy.lib.financial import ipmt
import pandas as pd
import cloudpickle

import matplotlib.pyplot as plt
from pathlib import Path
from utils.utils import get_dataframe_summary, process_to_len, class_from_path
import tqdm

class ProcessApplicationDataset(object):
    def __init__(self, config={}):
        self.config = {
            'final_data_dir': "",
            'save_dir': "",
            # key is column name in the dataframe, value is normalized range (to [0, value])
            'normalize_elements': {'ego_current_vel': 1, 'ego_current_steering': 2},
            'train_val_split_filter': None,
            'kept_columns': []
        }
        self.config.update(config)

        if self.config['train_val_split_filter'] is None:
            raise ValueError('train val split filter not provided')
        self.train_val_split_filter = class_from_path(self.config['train_val_split_filter'])
        
        self.final_data_fn = [str(p) for p in Path(self.config['final_data_dir']).rglob('*.pkl')]

        df_list = []
        for p in tqdm.tqdm(self.final_data_fn):
            df = pd.read_pickle(p)
            df_list.append(df)
        
        self.data = pd.concat(df_list)
        print(get_dataframe_summary(self.data))


    def normalize(self, df, normalize_elements={'ego_current_vel': 1, 'ego_current_steering': 2}):
        print(f"normalize_elements: {normalize_elements}")
        element_min_max = {}
        for k, v in normalize_elements.items():
            df['origin_'+k] = df[k]
            element = np.array(df[k].tolist())
            element_min = element.min()
            element_max = element.max()
            element_min_max[k] = {'min': element_min, 'max': element_max, 'upper_bound':v}
            df = df.apply(lambda x: v*(x-element_min)/(element_max-element_min) if x.name == k else x)

        # save image mean and variance for normalization #
        raster = np.array(df.current_ego_raster_img.tolist())
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

        df = pd.DataFrame(self.data)
        train_df, val_df = self.create_train_val_split(df)
        print(f"train_df shape is {train_df.shape}")
        print(f"val_df shape is {val_df.shape}")
        
        train_df.to_pickle(self.config['save_dir']+"/train.pkl")
        val_df.to_pickle(self.config['save_dir']+"/val.pkl")

    def create_train_val_split(self, df):
        normalized_df = self.normalize(df, normalize_elements=self.config['normalize_elements'])
        train_df, val_df = self.train_val_split_filter(normalized_df)

        normalized_df.to_pickle(self.config['save_dir']+"/full.pkl")

        return train_df[self.config['kept_columns']], val_df[self.config['kept_columns']]

        

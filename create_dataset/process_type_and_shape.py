import os
import pandas as pd
import tqdm
from utils.utils import process_to_len
import numpy as np

def process_once(data_df_list=[], data_save_dir=None, config={}):
    """ create dataset with the right obs_steps, pred_steps, etc
    
    :param data_df_list: list of scene names to process
    :param data_save_dir: directory to save processed raw data
    :param config: additional configs
    :returns: one pandas dataframe for each scene

    """

    obs_steps = config['other_configs']['obs_steps']
    pred_steps = config['other_configs']['pred_steps']
    nb_closest_neighbors = config['other_configs']['nb_closest_neighbors']

    numpy_data_keys = ['pos', 'quat', 'accel', 'speed', 'steering', 'raster']
    
    for df_fn in tqdm.tqdm(data_df_list):
        df = pd.read_pickle(df_fn)
        scene_name = df.iloc[0].scene_name
        
        training_df_dict = {}
        for k in list(df.keys()):
            training_df_dict[k] = []

        for i, r in df.iterrows():
            for k in list(r.keys()):
                for np_k in numpy_data_keys:
                    if np_k not in k:
                        training_df_dict[k].append(r[k])
                    else:
                        if 'current' in k:
                            training_df_dict[k].append(np.array(r[k]))
                        elif 'past' in k:
                            if r[k] == []:
                                r[k] = np.zeros(np.array(r['current'+k[4:]]).shape)[np.newaxis]

                            processed_past = process_to_len(np.array(r[k]), obs_steps, name=k, dim=0, before_or_after='before', mode='constant')
                            training_df_dict[k].append(processed_past)
                        elif 'future' in k:
                            if r[k] == []:
                                r[k] = np.zeros(np.array(r['current'+k[6:]]).shape)[np.newaxis]
                            processed_future = process_to_len(np.array(r[k]), pred_steps, name=k, dim=0, before_or_after='after', mode='constant')
                            training_df_dict[k].append(processed_future)

        training_df = pd.DataFrame(training_df_dict)

        import ipdb; ipdb.set_trace()
        training_df.to_pickle(data_save_dir+'/'+scene_name+".pkl")

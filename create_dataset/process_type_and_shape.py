import os
import pandas as pd
import tqdm
from utils.utils import process_to_len
import numpy as np

NO_CANBUS_SCENES = ['scene-0'+str(s) for s in [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]]


def process_once(data_df_list=[], data_save_dir=None, config={}):
    """ create dataset with the right obs_steps, pred_steps, etc
    
    :param data_df_list: list of scene names to process
    :param data_save_dir: directory to save processed raw data
    :param config: additional configs
    :returns: one pandas dataframe for each scene

    """

    obs_steps = config['other_configs']['obs_steps']
    pred_steps = config['other_configs']['pred_steps']

    numpy_data_keys = [
        'pos', 'quat', 'accel',
        'speed', 'steering', 'raster'
    ]
    
    for df_fn in tqdm.tqdm(data_df_list):
        df = pd.read_pickle(df_fn)
        scene_name = df.iloc[0].scene_name

        ######################################
        # remove scenes without CAN bus data #
        ######################################
        if scene_name in NO_CANBUS_SCENES:
            continue
        
        training_df_dict = {}
        for k in list(df.keys()):
            training_df_dict[k] = []
            
        for i, r in df.iterrows():
            for k in list(r.keys()):
                for np_k in numpy_data_keys:
                    if np_k not in k and 'current' not in k and 'past' not in k and 'future' not in k:
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

        ##################################
        # remove instance with NaN terms #
        ##################################
        del_row_idx = []
        for i, r in training_df.iterrows():
            for k in list(r.keys()):
                for np_k in numpy_data_keys:
                    if np_k in k:
                        if np.isnan((r[k]).mean()):
                            del_row_idx.append(i)

        training_df = training_df.drop(del_row_idx).reset_index(drop=True)
        training_df.to_pickle(data_save_dir+'/'+scene_name+".pkl")

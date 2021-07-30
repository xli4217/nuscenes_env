import tqdm
import pandas as pd
from utils.utils import process_to_len
import numpy as np

def process_once(df_path_list=[], data_save_dir=None, config={}):
    nb_closest_ado = config['other_configs']['nb_closest_ado']
    obs_steps = config['other_configs']['obs_steps']
    pred_steps = config['other_configs']['pred_steps']

    maneuver2idx = {
        'turn_left': 1,
        'turn_right': 2
    }
    
    for scene_path in tqdm.tqdm(df_path_list, 'processing scene'):
        df = pd.read_pickle(scene_path)

        df_keys = list(df.keys())
                
        agn_df_dict = {}
        for k in df_keys:
            agn_df_dict[k] = []

        pad_mode = 'constant'
        for i, r in tqdm.tqdm(df.iterrows()):
            for k in df_keys:
                processed_r_k = None
                ### process scene info ####
                if k == 'scene_name':
                    assert isinstance(r[k], str)
                elif k == 'sample_token':
                    assert isinstance(r[k], str)
                elif k == 'sample_idx':
                    assert isinstance(r[k], int)

                #### process ego info ####
                elif k == 'ego_current':
                    assert r[k].shape == (pred_steps,)
                elif k == 'ego_future':
                    processed_r_k = process_to_len(r[k], pred_steps, name=k, dim=0, before_or_after='after', mode=pad_mode)
                    assert processed_r_k.shape == (pred_steps,2)
                elif k == "ego_past":
                    processed_r_k = process_to_len(r[k], obs_steps, name=k, dim=0, before_or_after='before', mode=pad_mode)
                    assert processed_r_k.shape == (obs_steps,2)
                elif k == 'ego_current_vel':
                    assert isinstance(r[k], float)
                elif k == 'ego_future_vel':
                    processed_r_k = process_to_len(r[k], pred_steps, name=k, dim=0, before_or_after='after', mode=pad_mode)
                    assert processed_r_k.shape == (pred_steps,)
                elif k == 'ego_past_vel':
                    processed_r_k = process_to_len(r[k], obs_steps, name=k, dim=0, before_or_after='before', mode=pad_mode)
                    assert processed_r_k.shape == (obs_steps,)
                elif k == 'ego_current_steering':
                    assert isinstance(r[k], float)
                elif k == 'ego_future_steering':
                    processed_r_k = process_to_len(r[k], pred_steps, name=k, dim=0, before_or_after='after', mode=pad_mode)
                    assert processed_r_k.shape == (pred_steps,)
                elif k == 'ego_past_steering':
                    processed_r_k = process_to_len(r[k], obs_steps, name=k, dim=0, before_or_after='before', mode=pad_mode)
                    assert processed_r_k.shape == (obs_steps,)
                elif k == 'ego_goal':
                    assert r[k].shape == (2, ), f"ego_goal shape is {r[k].shape}"
                elif k == 'current_ego_raster_img':
                    assert r[k].shape == (3, 250, 250)
                elif k == 'future_ego_raster_img':
                    processed_r_k = process_to_len(r[k], pred_steps, name=k, dim=0, before_or_after='after', mode=pad_mode)
                    assert processed_r_k.shape == (pred_steps, 3, 250, 250)
                elif k == 'past_ego_raster_img':
                    if r[k].size == 0:
                        processed_r_k = process_to_len(r['current_ego_raster_img'][np.newaxis], obs_steps, name=k, dim=0, before_or_after='before', mode=pad_mode)
                    else:
                        processed_r_k = process_to_len(r[k], obs_steps, name=k, dim=0, before_or_after='before', mode=pad_mode)
                    assert processed_r_k.shape == (obs_steps, 3, 250, 250), f"actual shape is {processed_r_k.shape}"
                elif k == 'current_ego_maneuvers':
                    if len(r[k]) == 0:
                        processed_r_k = 0
                    else:
                        processed_r_k = maneuver2idx[r[k][0]]
                elif k == 'future_ego_maneuvers':
                    numeric_maneuvers = []
                    for m in r[k]:
                        if len(m) == 0:
                            numeric_maneuvers.append(0)
                        else:
                            numeric_maneuvers.append(maneuver2idx[m[0]])
                    processed_r_k = process_to_len(np.array(numeric_maneuvers), pred_steps, name=k, dim=0, before_or_after='after', mode=pad_mode)
                elif k == 'past_ego_maneuvers':
                    numeric_maneuvers = []
                    for m in r[k]:
                        if len(m) == 0:
                            numeric_maneuvers.append(0)
                        else:
                            numeric_maneuvers.append(maneuver2idx[m[0]])
                    if len(numeric_maneuvers) == 0:
                        numeric_maneuvers = [0]
                    processed_r_k = process_to_len(np.array(numeric_maneuvers), obs_steps, name=k, dim=0, before_or_after='before', mode=pad_mode)

                # TODO: road objects and interactions refer to gnn
                elif k == ' ego_road_objects':
                    assert isinstance(r[k], list)
                elif k == 'ego_interactions':
                    assert isinstance(r[k], list)
                elif k == 'discretized_lane':
                    # TODO: this takes the first lane, not always the lane human ego takes
                    processed_r_k = process_to_len(r[k][0], 200, name='discretized_lane', dim=0, before_or_after='after', mode=pad_mode)
                    assert processed_r_k.shape == (200,2)

                #### process ado info ####
                elif k == 'ado_dict':
                    current_pos = []
                    current_vel = []
                    token = []
                    future = []
                    past = []
                    road_objects = []
                    interactions = []

                    for ado_token, ado_info in r[k].items():
                        for k1, v in ado_info.items():
                            if k1 == 'current_pos':
                                processed_r_k = None
                            if k1 == 'current_vel':
                                processed_r_k = None
                            if k1 == 'token':
                                processed_r_k = None
                            if k1 == 'future':
                                processed_r_k = None
                            if k1 == 'past':
                                processed_r_k = None
                            if k1 == 'current_interactions':
                                processed_r_k = None
                        
               
                if processed_r_k is None:
                    agn_df_dict[k].append(r[k])
                else:
                    agn_df_dict[k].append(processed_r_k)

        scene_df = pd.DataFrame(agn_df_dict)
        scene_df.to_pickle(data_save_dir+"/"+df.iloc[0].scene_name+".pkl")

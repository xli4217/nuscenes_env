import pandas as pd
from utils.utils import assert_shape, process_to_len
import tqdm
import numpy as np

def add_row(df_dict, r, sample_df, scene_name, sample_idx, ego_or_ado='ado', nb_closest_neighbors=4, max_neigbhor_range=40):
    if ego_or_ado == 'ego':
        name = 'ego'
        token = 'ego'
    else:
        name = 'instance'
        token = r.instance_token


    current_neighbor_tokens = []

    current_neighbor_pos = []
    past_neighbor_pos = []
    future_neighbor_pos = []

    current_neighbor_speed = []
    past_neighbor_speed = []
    future_neighbor_speed = []
    
    for token, dist in zip(r['current_'+name+'_neighbors'][0], r['current_'+name+'_neighbors'][1]):
        if dist < max_neigbhor_range:
            if token == 'ego':
                r1 = sample_df.iloc[0]
                r1_name = 'ego'
            else:
                r1 = sample_df.loc[sample_df.instance_token==token]
                if r1.shape[0] == 0:
                    continue
                r1 = r1.iloc[0]
                r1_name = 'instance'
                
            current_neighbor_tokens.append(token)

            current_neighbor_pos.append(r1['current_'+r1_name+'_pos'])
            past_neighbor_pos.append(r1['past_'+r1_name+'_pos'])
            future_neighbor_pos.append(r1['future_'+r1_name+'_pos'])

            if r1_name == 'ego':
                current_speed = r1['current_'+r1_name+'_speed'][0]
                past_speed = r1['past_'+r1_name+'_speed'][:,0]
                future_speed = r1['future_'+r1_name+'_speed'][:,0]
            else:
                current_speed = r1['current_'+r1_name+'_speed']
                past_speed = r1['past_'+r1_name+'_speed']
                future_speed = r1['future_'+r1_name+'_speed']
     
            current_neighbor_speed.append(current_speed)
            past_neighbor_speed.append(past_speed)
            future_neighbor_speed.append(future_speed)


    if len(current_neighbor_pos) == 0:
        return df_dict

    current_neighbor_pos = np.array(current_neighbor_pos)
    current_neighbor_pos = process_to_len(current_neighbor_pos, nb_closest_neighbors, 'current_neighbor_pos')
    
    past_neighbor_pos = np.array(past_neighbor_pos)
    past_neighbor_pos = process_to_len(past_neighbor_pos, nb_closest_neighbors, 'past_neighbor_pos')
    future_neighbor_pos = np.array(future_neighbor_pos)
    future_neighbor_pos = process_to_len(future_neighbor_pos, nb_closest_neighbors, 'future_neighbor_pos')

    current_neighbor_tokens = process_to_len(np.array(current_neighbor_tokens), nb_closest_neighbors, 'current_neighbor_tokens').tolist()
    
    current_neighbor_speed = np.array(current_neighbor_speed)
    current_neighbor_speed = process_to_len(current_neighbor_speed, nb_closest_neighbors, 'current_neighbor_speed')
    past_neighbor_speed = np.array(past_neighbor_speed)
    past_neighbor_speed = process_to_len(past_neighbor_speed, nb_closest_neighbors, 'past_neighbor_speed')
    future_neighbor_speed = np.array(future_neighbor_speed)
    future_neighbor_speed = process_to_len(future_neighbor_speed, nb_closest_neighbors, 'future_neighbor_speed')

    
    ## populate ##
    #### scene info ####
    df_dict['scene_name'] += [scene_name]
    df_dict['scene_token'] += [r.scene_token]
    df_dict['sample_idx'] += [sample_idx]
    df_dict['sample_token'] += [r.sample_token]

    #### agent info ####
    df_dict['agent_token'].append(token)

    df_dict['current_agent_pos'].append(r['current_'+name+'_pos'])
    df_dict['past_agent_pos'].append(r['past_'+name+'_pos'])
    df_dict['future_agent_pos'].append(r['future_'+name+'_pos'])

    df_dict['current_agent_speed'].append(r['current_'+name+'_speed'])
    df_dict['past_agent_speed'].append(r['past_'+name+'_speed'])
    df_dict['future_agent_speed'].append(r['future_'+name+'_speed'])
    
    df_dict['current_agent_raster_path'].append(r['current_'+name+'_raster_img_path'])
    df_dict['past_agent_raster_path'].append(r['past_'+name+'_raster_img_path'])
    df_dict['future_agent_raster_path'].append(r['future_'+name+'_raster_img_path'])

    df_dict['current_neighbor_tokens'].append(current_neighbor_tokens)
    
    df_dict['current_neighbor_pos'].append(current_neighbor_pos)
    df_dict['past_neighbor_pos'].append(past_neighbor_pos)
    df_dict['future_neighbor_pos'].append(future_neighbor_pos)

    df_dict['current_neighbor_speed'].append(current_neighbor_speed)
    df_dict['past_neighbor_speed'].append(past_neighbor_speed)
    df_dict['future_neighbor_speed'].append(future_neighbor_speed)

    
    return df_dict
    
def final_data_processor(df, config={}):
    nb_closest_neighbors = config['other_configs']['nb_closest_neighbors']
    max_neigbhor_range = config['other_configs']['max_neighbor_range']
    
    df_dict = {
        'scene_name': [],            # str
        'scene_token': [],           # str
        'sample_idx': [],            # int
        'sample_token': [],          # str
        'agent_token': [],           # str

        'current_agent_pos':[],      # np.ndarray (2,)
        'past_agent_pos': [],        # np.ndarray(obs_steps, 2)
        'future_agent_pos': [],      # np.ndarray(pred_steps, 2)

        'current_agent_speed':[],      # np.ndarray (1,)
        'past_agent_speed': [],        # np.ndarray(obs_steps, )
        'future_agent_speed': [],      # np.ndarray(pred_steps, )

        'current_agent_raster_path':[],   # np.ndarray(3, 250, 250)
        'past_agent_raster_path':[],      # np.ndarray(obs_steps, 3, 250, 250)
        'future_agent_raster_path':[],    # np.ndarray(pred_steps, 3, 250, 250)

        'current_neighbor_tokens':[], # list (nbr_neighbors)
        
        'current_neighbor_pos': [],  # np.ndarray(nbr_neighbors, 2)
        'future_neighbor_pos': [],   # np.ndarray(nbr_neighbors, pred_steps, 2)
        'past_neighbor_pos': [],     # np.ndarray(nbr_neighbors, obs_steps, 2)

        'current_neighbor_speed': [],  # np.ndarray(nbr_neighbors, )
        'future_neighbor_speed': [],   # np.ndarray(nbr_neighbors, pred_steps)
        'past_neighbor_speed': []     # np.ndarray(nbr_neighbors, obs_steps)

    }
    
    multi_scene_df = df.set_index(['scene_name', 'sample_idx'])
    #### loop through scenes ####
    for scene_name, scene_df in tqdm.tqdm(multi_scene_df.groupby(level=0)):
        print(f"processing scene {scene_name}")
        for sample_idx, sample_df in scene_df.groupby(level='sample_idx'):
            df_dict = add_row(df_dict,
                              sample_df.iloc[0],
                              sample_df, 
                              scene_name=scene_name,
                              sample_idx=sample_idx,
                              ego_or_ado='ego',
                              nb_closest_neighbors=nb_closest_neighbors,
                              max_neigbhor_range=max_neigbhor_range)
            
            for i, r in sample_df.iterrows():
                df_dict = add_row(df_dict,
                                  r,
                                  sample_df,
                                  scene_name=scene_name,
                                  sample_idx=sample_idx,
                                  ego_or_ado='ado',
                                  nb_closest_neighbors=nb_closest_neighbors,
                                  max_neigbhor_range=max_neigbhor_range)
            
    df = pd.DataFrame(df_dict)
    
    return df

def train_val_split_filter(df, config={}):
    train_df = df.iloc[:int(0.7*df.shape[0])]
    val_df = df.iloc[int(0.7*df.shape[0]):]

    return train_df, val_df
    # df = df.sample(frac=1).reset_index(drop=True)
        
    # straight_df = []
    # left_df = []
    # right_df = []

    # for i, r in df.iterrows():
    #     if r.current_ego_maneuvers == 0:
    #         straight_df.append(r)
    #     if r.current_ego_maneuvers == 1:
    #         left_df.append(r)
    #     if r.current_ego_maneuvers == 2:
    #         right_df.append(r)

    # min_nbr_rows = min(len(straight_df), len(left_df), len(right_df))

    # straight_df = pd.DataFrame(straight_df).iloc[:min_nbr_rows]
    # left_df = pd.DataFrame(left_df).iloc[:min_nbr_rows]
    # right_df = pd.DataFrame(right_df)[:min_nbr_rows]

    # print(f"straight size is {straight_df.shape}")
    # print(f"left size is {left_df.shape}")
    # print(f"right size is {right_df.shape}")
    # print(f"taken size is {min_nbr_rows}")

    # train_straight_df = straight_df.iloc[:int(0.7*straight_df.shape[0])]
    # val_straight_df = straight_df.iloc[int(0.7*straight_df.shape[0]):]
    
    # train_left_df = left_df.iloc[:int(0.7*left_df.shape[0])]
    # val_left_df = left_df.iloc[int(0.7*left_df.shape[0]):]
    
    # train_right_df = right_df.iloc[:int(0.7*right_df.shape[0])]
    # val_right_df = right_df.iloc[int(0.7*right_df.shape[0]):]

    # train_df = pd.concat([train_straight_df, train_left_df, train_right_df])
    # val_df = pd.concat([val_straight_df, val_left_df, val_right_df])

        
    # return train_df, val_df



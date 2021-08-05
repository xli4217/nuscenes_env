import pandas as pd
from utils.utils import assert_shape
import tqdm

def add_row(df_dict, r, scene_name, sample_idx, ego_or_ado='ado'):
    if ego_or_ado == 'ego':
        name = 'ego'
        token = 'ego'
    else:
        name = 'instance'
        token = r.instance_token

    df_dict['scene_name'] += [scene_name]
    df_dict['scene_token'] += [r.scene_token]
    df_dict['sample_idx'] += [sample_idx]
    df_dict['sample_token'] += [r.sample_token]

    df_dict['agent_token'].append(token)
    df_dict['current_agent_pos'].append(r['current_'+name+'_pos'])
    df_dict['past_agent_pos'].append(r['past_'+name+'_pos'])
    df_dict['future_agent_pos'].append(r['future_'+name+'_pos'])
    df_dict['current_agent_raster'].append(r['current_'+name+'_raster_img'])
    df_dict['past_agent_raster'].append(r['past_'+name+'_raster_img'])
    df_dict['future_agent_raster'].append(r['future_'+name+'_raster_img'])

    return df_dict
    
def data_processor(df, config={}):
    nb_closest_neighbors = config['nb_closest_neighbors']

    df_dict = {
        'scene_name': [],           # str
        'scene_token': [],          # str
        'sample_idx': [],           # int
        'sample_token': [],         # str
        'agent_token': [],          # str
        'current_agent_pos':[],     # np.ndarray (2,)
        'past_agent_pos': [],       # np.ndarray(obs_steps, 2)
        'future_agent_pos': [],     # np.ndarray(pred_steps, 2)
        'current_agent_raster':[],  # np.ndarray(3, 250, 250)
        'past_agent_raster':[],     # np.ndarray(obs_steps, 3, 250, 250)
        'future_agent_raster':[],   # np.ndarray(pred_steps, 3, 250, 250)
        'current_neighbor_pos': [], # np.ndarray(nbr_neighbors, 2)
        'future_neighbor_pos': [],  # np.ndarray(nbr_neighbors, pred_steps, 2)
        'past_neighbor_pos': [],    # np.ndarray(nbr_neighbors, obs_steps, 2)
        'current_neighbor_tokens':[]# list (nbr_neighbors)
    }
    
    multi_scene_df = df.set_index(['scene_name', 'sample_idx'])
    #### loop through scenes ####
    for scene_name, scene_df in tqdm.tqdm(multi_scene_df.groupby(level=0)):
        print(f"processing scene {scene_name}")
        for sample_idx, sample_df in scene_df.groupby(level='sample_idx'):
            df_dict = add_row(df_dict, sample_df.iloc[0], scene_name=scene_name, sample_idx=sample_idx, ego_or_ado='ego')

            for i, r in sample_df.iterrows():
                df_dict = add_row(df_dict, r, scene_name=scene_name, sample_idx=sample_idx, ego_or_ado='ado')
            
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



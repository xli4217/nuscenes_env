import ipdb
import pandas as pd
from utils.utils import process_to_len, populate_dictionary
import tqdm
import numpy as np
from utils.utils import convert_local_coords_to_global, convert_global_coords_to_local

def add_row(df_dict, r, sample_df, scene_name, sample_idx, ego_or_ado='ado', nb_closest_neighbors=4, max_neigbhor_range=40):
    if ego_or_ado == 'ego':
        name = 'ego'
        token = 'ego'
    else:
        name = 'instance'
        token = r.instance_token

    #### filter data ####
    violating_conditions = [
        # no future data for agent
        np.linalg.norm(r['future_'+name+'_pos']) < 0.01
    ]
    if any(violating_conditions):
        return df_dict

    obs_steps = r.past_ego_pos.shape[0]
    pred_steps = r.future_ego_pos.shape[0]
        
    current_neighbor_tokens = []

    current_neighbor_pos = []
    past_neighbor_pos = []
    future_neighbor_pos = []

    current_neighbor_speed = []
    past_neighbor_speed = []
    future_neighbor_speed = []
    
    for atoken, dist in zip(r['current_'+name+'_neighbors'][0], r['current_'+name+'_neighbors'][1]):
        if dist < max_neigbhor_range:
            if atoken == 'ego':
                r1 = sample_df.iloc[0]
                r1_name = 'ego'
            else:
                r1 = sample_df.loc[sample_df.instance_token==atoken]
                if r1.shape[0] == 0:
                    continue
                r1 = r1.iloc[0]
                r1_name = 'instance'
                
            current_neighbor_tokens.append(atoken)

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
    df_dict['scene_name'].append(str(scene_name))
    df_dict['scene_token'].append(str(r.scene_token))
    df_dict['sample_idx'].append(int(sample_idx))
    df_dict['sample_token'].append(str(r.sample_token))

    # #### agent info ####
    df_dict['agent_token'].append(token)

    df_dict = populate_dictionary(df_dict, 'current_agent_pos', r['current_'+name+'_pos'][:2], np.ndarray, (2,), populate_func='append')    
    df_dict = populate_dictionary(df_dict, 'past_agent_pos', r['past_'+name+'_pos'][:,:2], np.ndarray, (obs_steps, 2), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'future_agent_pos', r['future_'+name+'_pos'][:,:2], np.ndarray, (pred_steps, 2), populate_func='append')

    df_dict = populate_dictionary(df_dict, 'current_agent_quat', r['current_'+name+'_quat'], np.ndarray, (4, ), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'past_agent_quat', r['past_'+name+'_quat'], np.ndarray, (obs_steps, 4), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'future_agent_quat', r['future_'+name+'_quat'], np.ndarray, (pred_steps, 4), populate_func='append')

    if name == 'ego':
        current_speed = np.array([r['current_'+name+'_speed'][0]])
        past_speed = r['past_'+name+'_speed'][:,0]
        future_speed = r['future_'+name+'_speed'][:,0]

        current_steering = np.array([r['current_'+name+'_steering'][-1]])
        past_steering = r['past_'+name+'_steering'][:,-1]
        future_steering = r['future_'+name+'_steering'][:,-1]
    else:
        current_speed = r['current_'+name+'_speed'][np.newaxis]
        past_speed = r['past_'+name+'_speed']
        future_speed = r['future_'+name+'_speed']

        current_steering = np.array([0])
        past_steering = np.zeros(obs_steps)
        future_steering = np.zeros(pred_steps)
        
    df_dict = populate_dictionary(df_dict, 'current_agent_speed', current_speed, np.ndarray, (1,), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'past_agent_speed', past_speed, np.ndarray, (obs_steps, ), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'future_agent_speed', future_speed, np.ndarray, (pred_steps, ), populate_func='append')

    df_dict = populate_dictionary(df_dict, 'current_agent_steering', current_steering, np.ndarray, (1,), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'past_agent_steering', past_steering, np.ndarray, (obs_steps, ), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'future_agent_steering', future_steering, np.ndarray, (pred_steps, ), populate_func='append')

    
    df_dict['current_agent_raster_path'].append(r['current_'+name+'_raster_img_path'])
    df_dict['past_agent_raster_path'].append(r['past_'+name+'_raster_img_path'])
    df_dict['future_agent_raster_path'].append(r['future_'+name+'_raster_img_path'])

    df_dict['current_neighbor_tokens'].append(current_neighbor_tokens)

    df_dict = populate_dictionary(df_dict, 'current_neighbor_pos', current_neighbor_pos[:,:2], np.ndarray, (nb_closest_neighbors, 2), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'past_neighbor_pos', past_neighbor_pos[:,:,:2], np.ndarray, (nb_closest_neighbors, obs_steps, 2), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'future_neighbor_pos', future_neighbor_pos[:,:,:2], np.ndarray, (nb_closest_neighbors, pred_steps, 2), populate_func='append')

    df_dict = populate_dictionary(df_dict, 'current_neighbor_speed', current_neighbor_speed, np.ndarray, (nb_closest_neighbors, ), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'past_neighbor_speed', past_neighbor_speed, np.ndarray, (nb_closest_neighbors, obs_steps), populate_func='append')
    df_dict = populate_dictionary(df_dict, 'future_neighbor_speed', future_neighbor_speed, np.ndarray, (nb_closest_neighbors, pred_steps), populate_func='append')

    
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

        'current_agent_quat':[],      # np.ndarray (4,)
        'past_agent_quat': [],        # np.ndarray(obs_steps, 4)
        'future_agent_quat': [],      # np.ndarray(pred_steps, 4)

        'current_agent_speed':[],      # np.ndarray (1,)
        'past_agent_speed': [],        # np.ndarray(obs_steps, )
        'future_agent_speed': [],      # np.ndarray(pred_steps, )

        'current_agent_steering':[],      # np.ndarray (1,)
        'past_agent_steering': [],        # np.ndarray(obs_steps, )
        'future_agent_steering': [],      # np.ndarray(pred_steps, )
        
        'current_agent_raster_path':[],   # str
        'past_agent_raster_path':[],      # list[str]
        'future_agent_raster_path':[],    # list[str]

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


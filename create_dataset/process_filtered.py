import os
import pandas as pd
import tqdm
import json
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from utils.utils import class_from_path
from .filters.interaction_filters import interaction_filter
from .filters.maneuver_filters import maneuver_filter
from .process_type_and_shape import process_once as process_type_and_shape_once
from .process_finalize import final_data_processor

def unique(list1, key):

    if isinstance(list1[0], list):
        list2 =[json.dumps(tp) for tp in list1]
    else:
        list2 = list1
        
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list2:
        if isinstance(list2[0], np.ndarray):
            in_list = False
            for y in unique_list:
                if np.array_equal(x,y):
                    in_list = True
            if not in_list:
                unique_list.append(x)
        else:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)

    if isinstance(unique_list[0], str) and isinstance(list1[0], list):
        r = [json.loads(tp) for tp in unique_list]
        return r

    return unique_list
    

def process_once(data_df_list=[], data_save_dir=None, config={}):
    """from each sample slice, add turn each agent field into past, current, future. And add filtered fields

    :param data_df_list: list of raw data dataframe file paths to process
    :param data_save_dir: directory to save processed raw data
    :param config: additional configs
    :returns: one pandas dataframe for each scene

    """

    keep_categories = config['other_configs']['categories']
    keep_attributes = config['other_configs']['attributes']
    keep_scenarios = config['other_configs']['scenarios']

    scenario_filter = config['other_configs']['scenario_filter']

    test_process_scene_name = config['scene_name']
    
    ##################################
    # add past and present to fields #
    ##################################
    for df_fn in tqdm.tqdm(data_df_list):
        #scene_name = df.iloc[0].scene_name
        scene_name = df_fn.split('/')[-1][:-4]

        if test_process_scene_name is not None:
            if not scene_name == test_process_scene_name:
                continue
        
        df = pd.read_pickle(df_fn)
        ego_traj_dict = {}
        instance_traj_dict = {}
        #### initialize filtered df dict ####
        filtered_df_dict = {}
        for key in list(df.keys()):
            if 'current' not in key:
                filtered_df_dict[key] = []
            else:
                filtered_df_dict[key] = []
                filtered_df_dict['past_'+key[8:]] = []
                filtered_df_dict['future_'+key[8:]] = []

                if 'ego' in key:
                    #### populate ego traj ####                    
                    ego_traj_dict[key[8:]+"_traj"] = unique(df[key].tolist(), key)

        ego_traj_dict['sample_token_traj'] = unique(df['sample_token'].tolist(), 'sample_token')
                    
       #### populate instance traj ####
        for instance_token in df.instance_token.unique().tolist():
            instance_df = df.loc[df.instance_token.str.contains(instance_token)]
            
            instance_tmp_dict = {}
            for k in list(instance_df.keys()):
                if 'current_instance' in k:
                    instance_tmp_dict[k[8:]+"_traj"] = instance_df[k].tolist()
                    instance_tmp_dict['sample_token_traj'] = unique(instance_df['sample_token'].tolist(), 'sample_token')
            instance_traj_dict[instance_token] = instance_tmp_dict


        
            
        for i, r in df.iterrows():
            # sample_idx = r.sample_idx
            for k in list(r.keys()):
                filtered_df_dict[k].append(r[k])
                #### popluate past and future ####
                if 'current_ego' in k:
                    sample_idx = ego_traj_dict['sample_token_traj'].index(r.sample_token)
                    filtered_df_dict['past_'+k[8:]].append(ego_traj_dict[k[8:]+"_traj"][:sample_idx])
                    filtered_df_dict['future_'+k[8:]].append(ego_traj_dict[k[8:]+"_traj"][sample_idx+1:])
                    
                if 'current_instance' in k:
                    sample_idx = instance_traj_dict[r.instance_token]['sample_token_traj'].index(r.sample_token)
                    past = instance_traj_dict[r.instance_token][k[8:]+"_traj"][:sample_idx]
                    future = instance_traj_dict[r.instance_token][k[8:]+"_traj"][sample_idx+1:]
                    filtered_df_dict['past_'+k[8:]].append(past)
                    filtered_df_dict['future_'+k[8:]].append(future)
                    #if r.sample_token == '0efac6cf63fd45aaa8c426f4ea7bd299' and r.instance_token == 'cec40f69a94740809e632323508f62d2':

                        
                        
        filtered_df = pd.DataFrame(filtered_df_dict)
        
        #########################
        # Add nearest neighbors #
        #########################
        ego_neighbors = []
        instance_neighbors = []
        
        sample_based_df = filtered_df.set_index(['sample_idx'])
        for sample_idx, sample_df in sample_based_df.groupby(level='sample_idx'):
            all_agent_tokens = sample_df.instance_token.tolist()
            all_agent_tokens.append('ego')
            all_agent_tokens = np.array(all_agent_tokens)
            all_agent_pos = sample_df.current_instance_pos.tolist()
            all_agent_pos.append(sample_df.iloc[0].current_ego_pos)
            all_agent_pos = np.array(all_agent_pos)
            
            #### get ego neighbors ####
            dist = np.linalg.norm(np.array(sample_df.iloc[0].current_ego_pos) - all_agent_pos, axis=1)
            idx = np.argsort(dist)
            sorted_all_agent_tokens = all_agent_tokens[idx]
            sorted_dist = dist[idx]
            ego_neighbors += [(sorted_all_agent_tokens.tolist()[1:], sorted_dist.tolist()[1:])] * sample_df.shape[0]

            #### get ado neighbors ####
            for i, r in sample_df.iterrows():
                dist = np.linalg.norm(np.array(r.current_instance_pos) - all_agent_pos, axis=1)
                idx = np.argsort(dist)
                sorted_all_agent_tokens = all_agent_tokens[idx]
                sorted_dist = dist[idx]
                instance_neighbors.append((sorted_all_agent_tokens.tolist()[1:], sorted_dist.tolist()[1:]))

        filtered_df['current_ego_neighbors'] = ego_neighbors
        filtered_df['current_instance_neighbors'] = instance_neighbors

        #############
        # Filtering #
        #############
        #### filter categories ####
        if len(keep_categories) > 0:
            filtered_df = filtered_df[filtered_df.instance_category.str.contains('|'.join(keep_categories))]
            if filtered_df is None:
                return None
            filtered_df.reset_index(drop=True, inplace=True)
            
        #### filter attributes ####
        
        ### filter scenarios ####
        if len(keep_scenarios) > 0:
            filtered_df = class_from_path(scenario_filter)(filtered_df, keep_scenarios)
            if filtered_df is None:
                return None
            filtered_df.reset_index(drop=True, inplace=True)
            
        ##########################
        # Process type and shape #
        ##########################
        filtered_df = process_type_and_shape_once([filtered_df], config=config)
        if filtered_df is None:
            return None
        filtered_df.reset_index(drop=True, inplace=True)

        ##########################
        # Normalize And Finalize #
        ##########################        
        # TODO: check raster img paths
        filtered_df = final_data_processor(filtered_df, config)
        if filtered_df is None:
            return None
        filtered_df.reset_index(drop=True, inplace=True)
        
        ##################################
        # Add Maneuvers And Interactions #
        ##################################
        #### add maneuvers ####
        filtered_df = maneuver_filter(filtered_df)
        filtered_df.reset_index(drop=True, inplace=True)
        
        #### add interactions ####
        filtered_df = interaction_filter(filtered_df)
        filtered_df.reset_index(drop=True, inplace=True)
        
        ##########################
        # save filtered_scene_df #
        ##########################
        filtered_df.to_pickle(data_save_dir+"/"+scene_name+".pkl")
        


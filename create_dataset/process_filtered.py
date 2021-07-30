import os
import pandas as pd
import tqdm
import json
import numpy as np

def unique(list1):

    if isinstance(list1[0], list):
        list1 =[json.dumps(tp) for tp in list1]
    
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if isinstance(list1[0], list):
            if x not in unique_list:
                unique_list.append(x)
        if isinstance(list1[0], np.ndarray):
            in_list = False
            for y in unique_list:
                if np.array_equal(x,y):
                    in_list = True
            if not in_list:
                unique_list.append(x)
                
    if isinstance(list1[0], list):
        return [json.dumps(tp) for tp in unique_list]

    return unique_list
            
def process_once(data_df_list=[], data_save_dir=None, config={}):
    """from each sample slice, add turn each agent field into past, current, future. And add filtered fields

    :param data_df_list: list of raw data dataframe file paths to process
    :param data_save_dir: directory to save processed raw data
    :param config: additional configs
    :returns: one pandas dataframe for each scene

    """

    for df_fn in tqdm.tqdm(data_df_list):
        df = pd.read_pickle(df_fn)
        scene_name = df.iloc[0].scene_name
        
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
                    ego_traj_dict[key[8:]+"_traj"] = unique(df[key].tolist())

        #### populate instance traj ####
        for instance_token in df.instance_token.unique().tolist():
            instance_df = df.loc[df.instance_token.str.contains(instance_token)]

            instance_tmp_dict = {}
            for k in list(instance_df.keys()):
                if 'current_instance' in k:
                    instance_tmp_dict[k[8:]+"_traj"] = instance_df[k].tolist()
                        
            instance_traj_dict[instance_token] = instance_tmp_dict

                
        for i, r in df.iterrows():
            sample_idx = r.sample_idx
            
            for k in list(r.keys()):
                filtered_df_dict[k].append(r[k])

                #### popluate past and future ####
                if 'current_ego' in k:
                    filtered_df_dict['past_'+k[8:]].append(ego_traj_dict[k[8:]+"_traj"][:sample_idx])
                    filtered_df_dict['future_'+k[8:]].append(ego_traj_dict[k[8:]+"_traj"][sample_idx+1:])
                if 'current_instance' in k:
                    filtered_df_dict['past_'+k[8:]].append(instance_traj_dict[r.instance_token][k[8:]+"_traj"][:sample_idx])
                    filtered_df_dict['future_'+k[8:]].append(instance_traj_dict[r.instance_token][k[8:]+"_traj"][sample_idx+1:])

        filtered_df = pd.DataFrame(filtered_df_dict)
        filtered_df.to_pickle(data_save_dir+"/"+scene_name+".pkl")
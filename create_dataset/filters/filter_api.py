import ray
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import tqdm
from create_dataset.vehicle_behavior_filter.filters.utils import construct_filter_input
from pathlib import Path
from utils.utils import split_list_for_multi_worker, timing_val, set_function_arguments, class_from_path


def construct_filter_inputs(df_row1=None, agent1_is_ado=True, df_row2=None, agent2_is_ado=True):
    agent_trajectories = OrderedDict()
    map_elements = OrderedDict()

    if agent1_is_ado:
        agent1_key = df_row1.instance_token
    else:
        agent1_key = 'ego'
    agent1_traj, agent1_map = construct_filter_input(df_row1, agent1_is_ado)
        
    if agent2_is_ado:
        agent2_key = df_row2.instance_token
    else:
        agent2_key = 'ego'
    agent2_traj, agent2_map = construct_filter_input(df_row2, agent2_is_ado)
            
    agent_trajectories[agent1_key] = agent1_traj
    agent_trajectories[agent2_key] = agent2_traj

    map_elements[agent1_key] = agent1_map
    map_elements[agent2_key] = agent2_map

    return agent_trajectories, map_elements

def run_once(df_path_list=[], save_dir=None, config={}):
    """Creates a pandas dataframe

    :returns: pandas dataframe with each row representing an agent that's in at least one of the scenarios, has at least one of the interactions  

    """
    df_list = []
    for p in df_path_list:
        df_list.append(pd.read_pickle(p))
    df = pd.concat(df_list)

    #### filter scenarios ####
    filtered_scene_list = []
    multi_scene_df = df.set_index(['scene_name', 'sample_idx'])
    #### loop through scenes ####
    for scene_name, scene_df in multi_scene_df.groupby(level=0):
        #### loop through each step in the scene ####
        for sample_idx, sample_df in scene_df.groupby(level='sample_idx'):
            for n, sf in config['other_configs']['scenario_filters'].items():
                if sf(sample_df.iloc[0].ego_road_objects):
                    filtered_scene_list.append(scene_name)                    
    filtered_scene_df = df.loc[df['scene_name'].isin(filtered_scene_list)]

    #### add interactions and maneuvers ####
    ado_interactions = []
    ado_maneuvers = []
    ego_interactions = []
    ego_maneuvers = []

    multi_scene_df = filtered_scene_df.set_index(['scene_name'])
    #### loop through scenes ####
    for scene_name, scene_df in tqdm.tqdm(multi_scene_df.groupby(level=0), f"scene {scene_name}"):
        scene_ado_interactions = []
        scene_ado_maneuvers = []
        scene_ego_interactions = []
        scene_ego_maneuvers = []
        print(f"processing scene {scene_name}")
        #### loop through each step in the scene ####
        for idx1, row1 in scene_df.iterrows():
            ado_interaction_list = []
            ado_maneuver_list = []
            ego_interaction_list = []
            ego_maneuver_list = []

            for idx2, row2 in scene_df.iterrows():
                #### ado ####
                if row1.instance_token != row2.instance_token and row1.sample_time == row2.sample_time:
                    agent_trajectories, map_elements = construct_filter_inputs(row1, True, row2, True)
                    ## add interactions ##
                    for interaction_name, interaction_filter in config['other_configs']['interaction_filters'].items():
                        if interaction_filter(agent_trajectories, map_elements):
                            ado_interaction_list.append((interaction_name, row2.instance_token))
                                
                    
                #### ego ####
                if row1.sample_time == row2.sample_time:
                    ## add ego interactions ##
                    agent_trajectories, map_elements = construct_filter_inputs(row1, False, row2, True)
                    for interaction_name, interaction_filter in config['other_configs']['interaction_filters'].items():
                        if interaction_filter(agent_trajectories, map_elements):
                            ego_interaction_list.append((interaction_name, row2.instance_token))


                    
            for road_object_name, road_object_filter in config['other_configs']['scenario_filters'].items():
                #### add ado road objects ####
                if road_object_filter(row1.instance_road_objects):
                    ado_interaction_list.append(('occupies', road_object_name))
                #### add ego road objects ####
                if road_object_filter(row1.ego_road_objects):
                    ego_interaction_list.append(('occupies', road_object_name))

            for maneuver_name, maneuver_filter in config['other_configs']['maneuver_filters'].items():
                ## add ado maneuvers ##
                ado_traj, ado_map = construct_filter_input(row1, ado=True)
                if maneuver_filter(ado_traj):
                    ado_maneuver_list.append(maneuver_name)
                
                ## add ego maneuvers ##
                ego_traj, ego_map = construct_filter_input(row1, ado=False)
                if maneuver_filter(ego_traj):
                    ego_maneuver_list.append(maneuver_name)
                    
            ado_interactions.append(ado_interaction_list)
            ado_maneuvers.append(ado_maneuver_list)
            ego_interactions.append(ego_interaction_list)
            ego_maneuvers.append(ego_maneuver_list)

            scene_ado_interactions.append(ado_interaction_list)
            scene_ado_maneuvers.append(ado_maneuver_list)
            scene_ego_interactions.append(ego_interaction_list)
            scene_ego_maneuvers.append(ego_maneuver_list)

        scene_df['instance_interactions'] = scene_ado_interactions
        scene_df['instance_maneuvers'] = scene_ado_maneuvers
        scene_df['ego_interactions'] = scene_ego_interactions
        scene_df['ego_maneuvers'] = scene_ego_maneuvers
        scene_df['scene_name'] = [scene_name] * len(scene_ado_interactions)
        
        if save_dir is not None:
            scene_df.to_pickle(os.path.join(save_dir, scene_name+".pkl"))
            
            
        
if __name__ == "__main__":
    pass

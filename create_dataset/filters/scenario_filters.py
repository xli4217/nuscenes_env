import numpy as np
import tqdm
import pandas as pd

from paths import mini_path, full_path
# from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# nusc = NuScenes(version='v1.0-mini', dataroot=mini_path, verbose=True)

def is_in_intersection(road_objects):
    if 'is_intersection' in list(road_objects.keys()):
        if road_objects['is_intersection']:
            return True
    return False


def scenario_filter(scene_df, senarios=[]):
    nusc_map = NuScenesMap(dataroot=full_path, map_name=filtered_df.iloc[0]['scene_location'])
            
    del_idx = []
    for i, idx in scene_df.iterrows():
        instance_road_object_traj = idx.current_instance_on_road_objects + idx.future_instance_on_road_objects
        for road_objects in instance_road_object_traj:
            if 'intersection' in senarios:
                if road_objects['road_segment'] is not "":
                    r = nusc_map.get('road_segment', ego_road_objects['road_segment'])
                    if not r['is_intersection']:
                        pass
    filtered_df = scene_df.drop(del_idx).reset_index(drop=True)
    return scenario_filter
    
# def is_in_intersection1(df):
#     """returns scenes where ego vehicle has passed through an intersection

#     :param df: pandas dataframe of raw data
#     :returns: a dataframe with filtered scene data

#     """
#     filtered_scene_list = []
#     multi_scene_df = df.set_index(['scene_name', 'sample_idx'])
#     #### loop through scenes ####
#     for scene_name, scene_df in tqdm.tqdm(multi_scene_df.groupby(level=0)):
#         print(f"processing scene {scene_name}")
#         #### loop through each step in the scene ####
#         for sample_idx, sample_df in scene_df.groupby(level='sample_idx'):
#             sample = nusc.get('sample', sample_df.sample_token.tolist()[0])
#             scene = nusc.get('scene', sample['scene_token'])
#             scene_log = nusc.get('log', scene['log_token'])
#             nusc_map = NuScenesMap(dataroot=mini_path, map_name=scene_log['location'])

#             road_segment_token = sample_df.ego_road_objects.tolist()[0]['road_segment']
#             if road_segment_token is not '':
#                 r = nusc_map.get('road_segment', road_segment_token)
#                 if r['is_intersection']:
#                     filtered_scene_list.append(scene_name)
#                     break


#     filtered_scene_df = df.loc[df['scene_name'].isin(filtered_scene_list)]

#     return filtered_scene_df

import numpy as np
import tqdm
import pandas as pd

#### TODO: remove this 
# from paths import mini_path
# from nuscenes.nuscenes import NuScenes
# from nuscenes.map_expansion.map_api import NuScenesMap

# nusc = NuScenes(version='v1.0-mini', dataroot=mini_path, verbose=True)

def is_in_intersection(road_objects):
    if 'is_intersection' in list(road_objects.keys()):
        if road_objects['is_intersection']:
            return True
    return False

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

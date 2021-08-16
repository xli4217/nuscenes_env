import numpy as np
import tqdm
import pandas as pd

from paths import mini_path, full_path
# from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


def scenario_filter(scene_df, keep_senarios=[]):
    nusc_map = NuScenesMap(dataroot=full_path, map_name=scene_df.iloc[0]['scene_location'])
            
    keep_instance_token_list = []
    processed_instance_token_list = []
    for i, r in scene_df.iterrows():
        if r.instance_token in processed_instance_token_list:
            continue
        processed_instance_token_list.append(r.instance_token)
        instance_road_object_traj = r.past_instance_on_road_objects + [r.current_instance_on_road_objects] + r.future_instance_on_road_objects
        for road_objects in instance_road_object_traj:
            # filter ados that passes through intersections #
            if 'intersection' in keep_senarios:
                if road_objects['road_segment'] is not "":
                    ro = nusc_map.get('road_segment', road_objects['road_segment'])
                    if ro['is_intersection']:
                        keep_instance_token_list.append(r.instance_token)
                        continue
                        
    #filtered_df = scene_df.drop(del_idx).reset_index(drop=True)
    filtered_df = scene_df[scene_df.instance_token.str.contains('|'.join(keep_instance_token_list))]
    return filtered_df
    

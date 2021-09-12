import os
import pandas as pd
import cloudpickle
from utils.utils import convert_global_coords_to_local
import tqdm
import numpy as np

def gnn_adapt_one_df_row(r):
    if not isinstance(r.current_neighbor_raster_path, list):
        return False
        
    agent_info = {
        r.agent_token[:5]: {
            'current_pos': r.current_agent_pos,
            'current_quat': r.current_agent_quat,
            'past_pos': r.past_agent_pos,
            'future_pos': r.future_agent_pos,
            'current_speed': r.current_agent_speed,
            'past_speed': r.past_agent_speed,
            'future_speed': r.future_agent_speed,
            'current_raster_path': str(r.current_agent_raster_path),
            'past_raster_path': [str(p) for p in r.past_agent_raster_path],
            'future_raster_path': [str(p) for p in r.future_agent_raster_path],
            'current_interactions': [],
            'past_interactions': [],
            'future_interactions': []
        }
    }

    for i, agent_token in enumerate(r.current_neighbor_tokens):
        agent_info[agent_token[:5]] = {
            'current_pos': r.current_neighbor_pos[i],
            'current_quat': r.current_neighbor_quat[i],
            'past_pos': r.past_neighbor_pos[i],
            'future_pos': r.future_neighbor_pos[i],
            'current_speed': r.current_neighbor_speed[i],
            'past_speed': r.past_neighbor_speed[i],
            'future_speed': r.future_neighbor_speed[i],
            'current_raster_path': str(r.current_neighbor_raster_path[i]),
            'past_raster_path': [str(p) for p in r.past_neighbor_raster_path[i]],
            'future_raster_path': [str(p) for p in r.future_neighbor_raster_path[i]],
            'current_interactions': [],
            'past_interactions': [],
            'future_interactions': []
        }

    for curr_interaction  in r.current_interactions[0]:
        if len(curr_interaction) > 0:
            agent_info[curr_interaction[0][:5]]['current_interactions'].append((curr_interaction[1], curr_interaction[2][:5]))

    past_interactions = {}
    for t, interaction_at_t in enumerate(r.past_interactions):
        tmp = {}
        for interaction in interaction_at_t:
            if len(interaction) > 0:
                if interaction[0][:5] not in list(tmp.keys()):
                    tmp[interaction[0][:5]] = [(interaction[1],interaction[2][:5], t)]
                else:
                    tmp[interaction[0][:5]].append((interaction[1],interaction[2][:5], t))
            else:
                if interaction[0][:5] not in list(tmp.keys()):
                    tmp[interaction[0][:5]] = [()]
                else:
                    tmp[interaction[0][:5]].append(())

                    
        for k, v in tmp.items():
            if k not in list(past_interactions.keys()):
                past_interactions[k] = [v]
            else:
                past_interactions[k].append(v)

    for k, v in past_interactions.items():
        agent_info[k]['past_interactions'] = v

    future_interactions = {}
    for t, interaction_at_t in enumerate(r.future_interactions):
        tmp = {}
        for interaction in interaction_at_t:
            if len(interaction) > 0:
                if interaction[0][:5] not in list(tmp.keys()):
                    tmp[interaction[0][:5]] = [(interaction[1],interaction[2][:5], t)]
                else:
                    tmp[interaction[0][:5]].append((interaction[1],interaction[2][:5], t))
            else:
                if interaction[0][:5] not in list(tmp.keys()):
                    tmp[interaction[0][:5]] = [()]
                else:
                    tmp[interaction[0][:5]].append(())

                        
        for k, v in tmp.items():
            if k not in list(future_interactions.keys()):
                future_interactions[k] = [v]
            else:
                future_interactions[k].append(v)

    for k, v in future_interactions.items():
        agent_info[k]['future_interactions'] = v

            
    info = {
        'scene_name': r.scene_name,
        'sample_idx': r.sample_idx,
        'sample_token': r.sample_token,
        'agent_info': agent_info
    }

    return info

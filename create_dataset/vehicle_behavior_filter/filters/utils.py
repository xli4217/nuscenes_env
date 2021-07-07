import pandas as pd
import os
import numpy as np


from pathlib import Path
from utils.utils import get_dataframe_summary
from nuscenes_env.graphics.scene_graphics import SceneGraphics
from configs.configs import na_config


def construct_filter_input(df_row, ado=False):
    if not ado:
        agent_traj = {
            'past': np.array(df_row.ego_past_pos),
            'current': np.array(df_row.ego_pos_traj[min(int(df_row.sample_idx), len(df_row.ego_pos_traj)-1)]),
            'future': np.array(df_row.ego_future_pos),
            'vel': df_row.ego_speed_traj[min(int(df_row.sample_idx), len(df_row.ego_speed_traj))-1]
        }

        agent_map = {}

    else:
        agent_traj = {
            'past': np.array(df_row.instance_past),
            'current': np.array(df_row.instance_pos),
            'future': np.array(df_row.instance_future),
            'vel': df_row.instance_vel
        }
        agent_map = {}


    return agent_traj, agent_map
        
def plot_text_box(ax, text_string:str, pos: np.ndarray, facecolor: str='wheat'):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(pos[0], pos[1], text_string, fontsize=10, bbox=props)

def analyze_data(self, df):
    pass

def visualize_sample(df, scene_name, sample_idx, scene_graphics=None, inst_token_to_name={}):
    if scene_graphics is None:
        scene_graphics = SceneGraphics(config={'NuScenesAgent_config':na_config})

    multi_scene_df = df.set_index(['scene_name', 'sample_idx'])

    v_sample_df = multi_scene_df.loc[(scene_name, sample_idx),:]

    di = v_sample_df.iloc[0]
    sample_token = di.sample_token

    #### generate base plot ####
    fig, ax = scene_graphics.plot_ego_scene(sample_token=sample_token)

    #### plot interaction textbox ####
    for i, r in v_sample_df.iterrows():
        ## plot ado name ##
        textbox_pos = np.array(r.instance_pos) + np.array([0,1.2])
        ado_name = inst_token_to_name[r.instance_token]
        plot_text_box(ax, ado_name, textbox_pos)

        ## plot interaction text ##
        h = 2.8
        for interaction in r.interactions:
            textbox_pos = np.array(r.instance_pos) + np.array([0,h])
            plot_text_box(ax,interaction[0]+" "+inst_token_to_name[interaction[1]], textbox_pos)
            h += 2.8

def visualize_scene(df, scene_name, nbr_samples=1, sample_idx=None, scene_graphics=None):
    multi_scene_df = df.set_index(['scene_name', 'sample_idx'])

    #### create instance_token to ado_name map ####
    scene_df = df.loc[df.scene_name == scene_name]
    inst_token_to_name = {}
    i = 0
    for idx, row in scene_df.iterrows():
        if row.instance_token not in list(inst_token_to_name.keys()):
            inst_token_to_name[row.instance_token] = str(i)
            i += 1

    if sample_idx is not None:
        visualize_sample(df, scene_name, sample_idx, scene_graphics, inst_token_to_name)
    else:
        for i in range(nbr_samples):
            visualize_sample(df, scene_name, i, scene_graphics, inst_token_to_name)

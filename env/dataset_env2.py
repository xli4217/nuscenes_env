import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from utils.utils import convert_local_coords_to_global, convert_global_coords_to_local, assert_type_and_shape
from utils.transformations import *

from celluloid import Camera
from graphics.nuscenes_agent import NuScenesAgent
from graphics.scene_graphics import SceneGraphics
import copy
import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from input_representation.static_layers import StaticLayerRasterizer
from input_representation.agents import AgentBoxesWithFadedHistory
from input_representation.interface import InputRepresentation
from input_representation.combinators import Rasterizer

from PIL import Image

from pathlib import Path
from .env_utils import *
from .env_render import render

import cloudpickle

class DatasetEnv2(object):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = {
            'SceneGraphics_config': {},
            'train_dataset_path': "",
            'val_dataset_path': "",
            'render_elements': [],
            'raster_dir': None,
            'render_type': 'image',
            'render_paper_ready': True,
            'patch_margin':30
        }
        self.config.update(config)

        #### load dataset ####
        self.dataset = cloudpickle.load(open(self.config['train_dataset_path'], 'rb')) + cloudpickle.load(open(self.config['val_dataset_path'], 'rb'))

        #### scene graphics ####
        graphics_config = copy.deepcopy(self.config['SceneGraphics_config'])
        graphics_config['NuScenesAgent_config'] = self.config['NuScenesAgent_config']
        graphics_config['load_dataset'] = False
        self.graphics = SceneGraphics(graphics_config, helper, py_logger, tb_logger)

        
    def reset(self, scene_name, instance='ego'):
        self.scene_name = scene_name
        self.unsorted_agent_data = []
        for d in self.dataset:
            if d['scene_name'] == scene_name:
                agent_info = d['agent_info']
                for agent_token, info in agent_info.items():
                    if instance in agent_token and np.linalg.norm(info['current_pos']) < 0.001:
                        self.unsorted_agent_data.append({'agent_info': agent_info, 'sample_idx': d['sample_idx'], 'sample_token': d['sample_token']})

        #### sort agent_data ####
        sorted_sample_idx = np.argsort(np.array([int(d['sample_idx']) for d in self.unsorted_agent_data]))

        self.agent_data = []
        for idx in sorted_sample_idx:
            self.agent_data.append(self.unsorted_agent_data[idx])

        self.current_agent_idx = 0

        return self.agent_data[0]
        
    def step(self):
        a = self.agent_data[self.current_agent_idx]

        current_raster = np.asarray(Image.open(os.path.join(self.config['raster_dir'], a['agent_info']['ego']['current_raster_path'])))

        all_info = {
            'sim_ego_raster_image': current_raster
        }

        render_info = {
            'sim_ego_quat_gb': a['agent_info']['ego']['current_quat_gb'],
            'sim_ego_pos_gb': a['agent_info']['ego']['current_pos_gb'],
            'ap_speed': None,
            'ap_steering': None,
            'scene_name': self.scene_name,
            'all_info': all_info,
            'sample_token': a['sample_token'],
            'instance_token': 'ego',
            'sample_idx': a['sample_idx'],
            'save_image_dir': None,
            'agent_info': a['agent_info']
        }

        self.current_agent_idx += 1

        done = False

        if self.current_agent_idx > len(self.agent_data):
            done = True

        fig, ax = self.render(render_info, save_img_dir=None)
        other = {'render_ax': ax}

        return a, done, other
        
    def render(self, render_info={}, save_img_dir=None):
        def plot_text_box(ax, text_string:str, pos: np.ndarray, facecolor: str='wheat'):
            props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
            ax.text(pos[0], pos[1], text_string, fontsize=10, bbox=props, zorder=800)
        
        fig, ax =  render(self.graphics, render_info, self.config)

        if 'groundtruth' in self.config['render_elements']:
            #####################
            # Plot GroundTruths #
            #####################
            sim_ego_pos = render_info['sim_ego_pos_gb']
            sim_ego_quat = render_info['sim_ego_quat_gb']
            #### plot neighbor connection lines ####
            for name, info in render_info['agent_info'].items():
                n_current = convert_local_coords_to_global(info['current_pos'], sim_ego_pos, sim_ego_quat)[0]
                n_past = convert_local_coords_to_global(info['past_pos'], sim_ego_pos, sim_ego_quat)
                n_future = convert_local_coords_to_global(info['future_pos'], sim_ego_pos, sim_ego_quat) 
                if name != 'ego':
                    # plot connection lines to n_current
                    ax.plot([sim_ego_pos[0], n_current[0]], [sim_ego_pos[1], n_current[1]], 'b-', linewidth=2, zorder=400)
                    # plot ado past #
                    ax.scatter(n_past[:,0], n_past[:,1], c='grey', s=20, zorder=400)
                    # plot ado future #
                    ax.scatter(n_future[:,0], n_future[:,1], c='yellow', s=20, zorder=400)
                else:
                    # plot ego past #
                    ax.scatter(n_past[:,0], n_past[:,1], c='grey', s=20, zorder=400)

                    # plot ego future #
                    ax.scatter(n_future[:,0], n_future[:,1], c='yellow', s=20, zorder=400)

        if 'token_labels' in self.config['render_elements']:
            #### plot ado instance tokens ####
            for name, info in render_info['agent_info'].items():
                n_current = convert_local_coords_to_global(info['current_pos'], sim_ego_pos, sim_ego_quat)[0]
                plot_text_box(ax, name, n_current[:2]+np.array([0,1.2]), facecolor='red')

        if 'interaction_labels' in self.config['render_elements']:
            #### plot interaction labels ####
            agent_height_dict = {}
            for name, info in render_info['agent_info'].items():
                agent_i_interactions = info['current_interactions']
                for interaction in agent_i_interactions:
                    interaction_name, a2_token = interaction
                    a1_token = name
                    if a1_token not in list(agent_height_dict.keys()):
                        agent_height_dict[a1_token] = np.array([0, 2.4])
                    else:
                        agent_height_dict[a1_token] += 2
                    a1_pos= convert_local_coords_to_global(info['current_pos'], sim_ego_pos, sim_ego_quat)[0] 
                    plot_text_box(ax, interaction_name+" "+a2_token[:4], a1_pos+agent_height_dict[a1_token])
                
                
        return fig, ax

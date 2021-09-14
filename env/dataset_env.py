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
from .sensing import Sensor
from graphics.scene_graphics import SceneGraphics
import copy
import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from input_representation.static_layers import StaticLayerRasterizer
from input_representation.agents import AgentBoxesWithFadedHistory
from input_representation.interface import InputRepresentation
from input_representation.combinators import Rasterizer

from pathlib import Path
from .env_utils import *
from .env_render import render

from task_specific.dataset_adapter import gnn_adapt_one_df_row

class NuScenesDatasetEnv(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = {
            'NuScenesAgent_config':{},
            'data_dir': None,
            'raster_dir': None,
            'SceneGraphics_config': {},
            'render_paper_ready': False,
            'render_type': [],
            'render_elements': ['sim_ego'], # can contain ['sensor_info', 'sim_ego', 'human_ego', 'control_plots']
            'patch_margin': 30,
            'save_image_dir': None,
            'control_mode': 'position'
        }

        self.config.update(config)
        super().__init__(config=self.config['NuScenesAgent_config'], helper=helper, py_logger=py_logger, tb_logger=tb_logger)
        
        #### Instantiate SceneGraphics ####
        graphics_config = copy.deepcopy(self.config['SceneGraphics_config'])
        graphics_config['NuScenesAgent_config'] = self.config['NuScenesAgent_config']
        graphics_config['load_dataset'] = False
        self.graphics = SceneGraphics(graphics_config, self.helper, self.py_logger, self.tb_logger)


        #### Initialize Rasterizer ####
        self.rasterizer = None
        if 'raster_image' in self.config['all_info_fields']:
            static_layer_rasterizer = StaticLayerRasterizer(self.helper, resolution=0.2)
            agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=2, resolution=0.2)
            self.rasterizer = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        
        #### available scenes ####
        self.scene_list = [str(p).split('/')[-1] for p in Path(self.config['data_dir']).rglob('*.pkl')]
        
        #### Initialize ####
        self.all_info = {
            #### dataset ego ####
            'ego_pos_gb': None,
            'ego_quat_gb': None,
            'ego_pos_traj': None,
            'ego_speed': None,
            'ego_raster_image': None,
            'ego_yaw_rate': None,
            'past_ego_pos': None,
            #### simulated ego ####
            'sim_ego_pos_gb': None,
            'sim_ego_quat_gb': None,
            'sim_ego_pos_traj': None,
            'sim_ego_speed': None,
            'sim_ego_raster_image': None,
            'sim_ego_yaw_rate': None
        }
        #self.reset()

    def set_adapt_one_row_func(self, func=None):
        self.adapt_one_row = func
        
    def update_all_info(self):
        self.sample_idx = self.instance_sample_idx_list[self.df_idx]
        self.sample_token = self.instance_sample_token_list[self.df_idx]

        self.update_row(self.instance_token, self.sample_idx)
        
        self.all_info['ego_pos_gb'] = self.r.current_agent_pos
        self.all_info['ego_quat_gb'] = self.r.current_agent_quat
        self.all_info['ego_pos_traj'] = np.vstack([self.r.past_agent_pos, self.r.current_agent_pos[np.newaxis], self.r.future_agent_pos])
        self.all_info['ego_speed'] = self.r.current_agent_speed
        self.all_info['past_ego_pos'] = self.r.past_agent_pos
        
        self.all_info['ego_raster_image'] = plt.imread(os.path.join(self.config['raster_dir'], str(self.r.current_agent_raster_path)))
        self.all_info['ego_yaw_rate'] = self.r.current_agent_steering
    
        self.all_info['sim_ego_pos_gb'] = self.sim_ego_pos_gb
        self.all_info['sim_ego_quat_gb'] = self.sim_ego_quat_gb
        self.all_info['sim_ego_speed'] = self.sim_ego_speed
        self.all_info['sim_ego_pos_traj'] = np.vstack([self.r.past_agent_pos, self.r.current_agent_pos[np.newaxis], self.r.future_agent_pos])
        self.all_info['gnn_data'] = gnn_adapt_one_df_row(self.r)

        if self.adapt_one_row is not None:
            self.all_info['predictor_data'] = self.adapt_one_row(self.r, self.config['raster_dir'], obs_len=4, pred_len=6)[0]
        
        sim_ego_pose = {
            'translation': self.sim_ego_pos_gb,
            'rotation': self.sim_ego_quat_gb
        }

        sim_ego_raster_img = self.rasterizer.make_input_representation(instance_token=None, sample_token=self.sample_token, ego=True, ego_pose=sim_ego_pose, include_history=False)

        #sim_ego_raster_img = np.transpose(sim_ego_raster_img, (2,0,1))

        self.all_info['sim_ego_raster_image'] = sim_ego_raster_img
        self.all_info['sim_ego_yaw_rate'] = self.sim_ego_yaw_rate
    
    def update_row(self, instance_token, sample_idx=None):
        # get the correct row
        self.r = self.instance_df[self.instance_df.sample_idx==sample_idx].iloc[0]
            
    def reset(self, scene_name=None, instance_token='ego', sample_token=None,  sample_idx=None):
        if scene_name is None:
            scene_idx = np.random.choice(len(self.scene_list))
            self.scene_name = self.scene_list[scene_idx][:-4]
        else:
            self.scene_name = scene_name
            
        self.scene_data = pd.read_pickle(os.path.join(self.config['data_dir'], self.scene_name+".pkl"))
        self.instance_token = instance_token
    
        self.instance_df = self.scene_data.loc[self.scene_data.agent_token==instance_token]
        self.instance_sample_idx_list = self.instance_df.sample_idx.tolist()
        if len(self.instance_sample_idx_list) == 0:
            return None
        self.instance_sample_token_list = self.instance_df.sample_token.tolist()

        if sample_idx is not None:
            self.df_idx = np.argmin(abs(np.array(self.instance_sample_idx_list) - sample_idx))
        else:
            self.df_idx = 0

        if 'current_agent_steering' not in list(self.scene_data.keys()):
            return None
            
        self.sample_idx = self.instance_sample_idx_list[self.df_idx]
        self.sample_token = self.instance_sample_token_list[self.df_idx]
        self.update_row(self.instance_token, self.sample_idx)
        
        self.sim_ego_pos_gb = self.r.current_agent_pos
        self.sim_ego_quat_gb = self.r.current_agent_quat
        self.sim_ego_speed = self.r.current_agent_speed
        self.sim_ego_yaw_rate = self.r.current_agent_steering
        self.sim_ego_yaw = quaternion_yaw(Quaternion(self.sim_ego_quat_gb))
        
        self.update_all_info()
        return self.get_observation()
        
    def get_observation(self):
        return self.all_info

    def render(self, render_info={}, save_img_dir=None):
        def plot_text_box(ax, text_string:str, pos: np.ndarray, facecolor: str='wheat'):
            props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
            ax.text(pos[0], pos[1], text_string, fontsize=10, bbox=props, zorder=800)
        
        render_info['sim_ego_quat_gb'] = self.all_info['sim_ego_quat_gb']
        render_info['sim_ego_pos_gb'] = self.all_info['sim_ego_pos_gb']
        render_info['ap_speed'] = None
        render_info['ap_steering'] = None
        render_info['ap_timesteps'] = None
        render_info['scene_name'] = self.scene_name
        render_info['all_info'] = self.all_info
        render_info['sample_token'] = self.r.sample_token
        render_info['instance_token'] = self.r.agent_token
        render_info['sample_idx'] = self.sample_idx
        render_info['save_image_dir'] = save_img_dir

        fig, ax =  render(self.graphics, render_info, self.config)

        if 'groundtruth' in self.config['render_elements']:
            #####################
            # Plot GroundTruths #
            #####################
            sim_ego_pos = self.all_info['sim_ego_pos_gb']
            #### plot neighbor connection lines ####
            all_pos = np.vstack([self.r.current_neighbor_pos, self.r.current_agent_pos])

            for p1 in all_pos:
                for p2 in all_pos:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=0.5,alpha=0.3, zorder=400)

            
            for n_token, n_past, n_current, n_future in zip(self.r.current_neighbor_tokens, self.r.past_neighbor_pos, self.r.current_neighbor_pos, self.r.future_neighbor_pos):
                # plot connection lines to n_current
                #ax.plot([sim_ego_pos[0], n_current[0]], [sim_ego_pos[1], n_current[1]], 'b-', linewidth=2, zorder=400)
                # plot ado past #
                ax.scatter(n_past[:,0], n_past[:,1], c='grey', s=20, zorder=800)
                # plot ado future #
                ax.scatter(n_future[:,0], n_future[:,1], c='yellow', s=20, zorder=800)

                # plot ado past #
                ax.plot(n_past[:,0], n_past[:,1], c='grey', zorder=800)
                # plot ado future #
                ax.plot(n_future[:,0], n_future[:,1], c='yellow', zorder=800)

                
            # plot ego past #
            ax.scatter(self.r.past_agent_pos[:,0], self.r.past_agent_pos[:,1], c='grey', s=20, zorder=800)

            # plot ego future #
            ax.scatter(self.r.future_agent_pos[:,0], self.r.future_agent_pos[:,1], c='yellow', s=20, zorder=800)

            # plot ego past #
            ax.plot(self.r.past_agent_pos[:,0], self.r.past_agent_pos[:,1], zorder=800)

            # plot ego future #
            ax.plot(self.r.future_agent_pos[:,0], self.r.future_agent_pos[:,1], zorder=800)
 
            
        if 'token_labels' in self.config['render_elements']:
            #### plot ado instance tokens ####
            plot_text_box(ax, self.r.agent_token[:4], self.r.current_agent_pos[:2]+np.array([0,1.5]), facecolor='red')
            for i, instance_token in enumerate(self.r.current_neighbor_tokens):
                plot_text_box(ax, instance_token[:4], self.r.current_neighbor_pos[i][:2]+np.array([0,1.5]), facecolor='red')        

        if 'interaction_labels' in self.config['render_elements']:
            #### plot interaction labels ####
            agent_height_dict = {}
            for agent_i_interactions in self.r.current_interactions:
                for interaction in agent_i_interactions:
                    a1_token, interaction_name, a2_token = interaction
                    if a1_token not in list(agent_height_dict.keys()):
                        agent_height_dict[a1_token] = np.array([0, 2.4])
                    else:
                        agent_height_dict[a1_token] += 2
                    if a1_token == 'ego':
                        a1_pos = self.r.current_agent_pos[:2]
                    else:
                        idx = self.r.current_neighbor_tokens.index(a1_token)
                        a1_pos = self.r.current_neighbor_pos[idx][:2]
                
                    plot_text_box(ax, interaction_name+" "+a2_token[:4], a1_pos+agent_height_dict[a1_token])
                
                
        return fig, ax

    def step(self, action=None, render_info={}, save_img_dir=None):
        if self.py_logger is not None:
            self.py_logger.debug(f"received action: {action}")

        #### render ####
        fig, ax = None, None
        if len(self.config['render_type']) > 0:
            fig, ax = self.render(render_info, save_img_dir)

        if action is None:
            self.sim_ego_pos_gb = self.all_info['ego_pos_gb']
            self.sim_ego_quat_gb = self.all_info['ego_quat_gb']
            self.sim_ego_speed = self.all_info['ego_speed']
            self.sim_ego_yaw_rate = self.all_info['ego_yaw_rate']


        if self.config['control_mode'] == 'position' and action is not None:
            self.sim_ego_pos_gb = action
            self.sim_ego_quat_gb = self.all_info['ego_quat_gb']
        
        if self.config['control_mode'] == 'kinematics' and action is not None:
            #### using a unicycle model ####
            self.sim_ego_speed = action[0]
            self.sim_ego_yaw_rate = action[1]

            dx_local = self.sim_ego_speed * np.array([np.cos(self.sim_ego_yaw), np.sin(self.sim_ego_yaw)]) * 0.5

            self.sim_ego_pos_gb += dx_local

            self.sim_ego_yaw += self.sim_ego_yaw_rate * 0.5
            q = Quaternion(axis=[0,0,1], degrees=np.rad2deg(self.sim_ego_yaw))
            self.sim_ego_quat_gb = [q[0], q[1], q[2], q[3]]


        self.df_idx += 1
        self.update_all_info()
        done = False
        if self.df_idx >= len(self.instance_sample_idx_list):
            done = True
        other = {
            'render_fig': fig,
            'render_ax': ax
        }
        return self.get_observation(), done, other

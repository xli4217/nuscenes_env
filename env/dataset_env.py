import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from typing import List, Tuple, Dict, Union
from PIL import Image

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction.helper import angle_of_rotation
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box


from utils.utils import assert_tensor, convert_local_coords_to_global, convert_global_coords_to_local, assert_tensor, get_dataframe_summary, calculate_steering
from utils.transformations import *

from utils.utils import transform_mesh2D, translate_mesh2D, rotate_mesh2D, process_to_len

from graphics.nuscenes_agent import NuScenesAgent
from graphics.scene_graphics import SceneGraphics
import copy

from input_representation.static_layers import StaticLayerRasterizer
from input_representation.agents import AgentBoxesWithFadedHistory
from input_representation.interface import InputRepresentation
from input_representation.combinators import Rasterizer

from pathlib import Path
from .env_utils import *
from .env_render import render

from task_specific.dataset_adapter import gnn_adapt_one_df_row
from rich.console import Console; console = Console(); print = console.print

from paths import *

class NuScenesDatasetEnv(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = {
            'NuScenesAgent_config':{},
            'data_dir': None,
            'full_data_path': None,
            'data_type': 'scene', # this can be 'train' or 'scene'
            'raster_dir': None,
            'SceneGraphics_config': {},
            'render_paper_ready': False,
            'render_type': [],
            'all_info_fields': ['raster_image'],
            'render_elements': ['sim_ego'],# can contain ['groundtruth','token_labels', 'interaction_labels ,'sim_ego', 'human_ego', 'control_plots', 'risk_map', 'lanes']
            'patch_margin': 30,
            'save_image_dir': None,
            'control_mode': 'position' # this can be 'position' or 'kinematics' or 'trajectory'
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
        self.all_info = {}
    
        self.full_data = None       
        if self.config['data_type'] == 'train':
            train_df = pd.read_pickle(self.config['data_dir'] + 'train.pkl')
            val_df = pd.read_pickle(self.config['data_dir'] + 'val.pkl')
            self.df = pd.concat([train_df, val_df])
            print(get_dataframe_summary(self.df))
            if self.config['full_data_path'] is not None:
                self.full_data = pd.read_pickle(self.config['full_data_path'])
                print(f'Loaded full data from {self.config["full_data_path"]}')
                print(f"full data shape {self.full_data.shape}")
            
    
    def set_adapt_one_row_func(self, func=None, *args, **kwargs):
        self.adapt_one_row = func
        
    def update_all_info(self):
        if self.df_idx >= len(self.instance_sample_idx_list)-1:
            return {}
        self.sample_idx = self.instance_sample_idx_list[self.df_idx]
        self.sample_token = self.instance_sample_token_list[self.df_idx]

        self.update_row(self.instance_token, self.sample_idx)
        
        #### Dataset Ego ####
        self.all_info['sample_idx'] = self.sample_idx
        self.all_info['ego_pos_gb'] = self.r.current_agent_pos
        self.all_info['ego_quat_gb'] = self.r.current_agent_quat
        self.all_info['ego_pos_traj'] = np.vstack([self.r.past_agent_pos, self.r.current_agent_pos[np.newaxis], self.r.future_agent_pos])
        self.all_info['ego_speed'] = self.r.current_agent_speed
        self.all_info['past_ego_pos'] = self.r.past_agent_pos
        
        self.all_info['ego_raster_image'] = plt.imread(os.path.join(self.config['raster_dir'], str(self.r.current_agent_raster_path)))
        self.all_info['ego_yaw_rate'] = self.r.current_agent_steering
        self.all_info['current_ego_neighbor_pos'] = self.r.current_neighbor_pos
        self.all_info['ego_future_lanes'] = get_future_lanes(self.map, 
                                                             self.r.current_agent_pos, 
                                                             self.r.current_agent_quat, 
                                                             frame='global',
                                                             ego_speed=4)
                
        #### Sim Ego ####
        self.all_info['sim_ego_pos_gb'] = self.sim_ego_pos_gb
        self.all_info['sim_ego_quat_gb'] = self.sim_ego_quat_gb
        sim_ego_yaw = Quaternion(self.sim_ego_quat_gb)
        self.all_info['sim_ego_yaw_rad'] = angle_of_rotation(quaternion_yaw(sim_ego_yaw))
        self.all_info['sim_ego_speed'] = self.sim_ego_speed
        self.all_info['sim_ego_pos_traj'] = np.vstack([self.r.past_agent_pos, self.r.current_agent_pos[np.newaxis], self.r.future_agent_pos])
        sim_ego_pose = {
            'translation': self.sim_ego_pos_gb,
            'rotation': self.sim_ego_quat_gb
        }

        if self.rasterizer is not None:
            sim_ego_raster_img = self.rasterizer.make_input_representation(instance_token=None, sample_token=self.sample_token, ego=True, ego_pose=sim_ego_pose, include_history=False)
            #sim_ego_raster_img = np.transpose(sim_ego_raster_img, (2,0,1))
            self.all_info['sim_ego_raster_image'] = sim_ego_raster_img
            
        self.all_info['sim_ego_yaw_rate'] = self.sim_ego_yaw_rate
        self.all_info['sim_ego_goal'] = self.sim_ego_goal
        self.all_info['current_sim_ego_neighbor_pos'] = self.r.current_neighbor_pos
        self.all_info['sim_ego_future_lanes'] = get_future_lanes(self.map, 
                                                                 self.sim_ego_pos_gb, 
                                                                 self.sim_ego_quat_gb, 
                                                                 frame='global',
                                                                 ego_speed=4)
                                                                 
        
        # histories #
        if self.rasterizer is not None:
            self.sim_ego_raster_image_history.append(sim_ego_raster_img)
            self.all_info['sim_ego_raster_image_history'] = self.sim_ego_raster_image_history
        self.sim_ego_pos_history.append(np.array(self.sim_ego_pos_gb))
        self.all_info['sim_ego_pos_history'] = self.sim_ego_pos_history
        self.sim_ego_quat_history.append(np.array(self.sim_ego_quat_gb))
        self.all_info['sim_ego_quat_history'] = self.sim_ego_quat_history
        self.sim_ego_speed_history.append(np.array(self.sim_ego_speed))
        self.all_info['sim_ego_speed_history'] = self.sim_ego_speed_history
        self.sim_ego_steering_history.append(np.array(self.sim_ego_yaw_rate))
        self.all_info['sim_ego_steering_history'] = self.sim_ego_steering_history
        #print(self.all_info['sim_ego_steering_history'])
        #print(self.all_info['sim_ego_speed_history'])
        #print("------")
        
        if self.adapt_one_row is not None:
            config = {
                'obs_len': 4,
                'pred_len':6,
                'raster_dir': self.config['raster_dir']
            }
            self.config.update(config)
            self.all_info['adapt_one_row_data'] = self.adapt_one_row(self.r, 
                                                                     self.config, 
                                                                     obs=self.all_info,
                                                                     full_df=self.full_data, 
                                                                     sim_ego_raster_img_history=self.sim_ego_raster_image_history)[0]
        
            
    def update_row(self, instance_token, sample_idx=None):
        # get the correct row
        self.r = self.instance_df[self.instance_df.sample_idx==sample_idx].iloc[0]
            
    def reset(self, 
              scene_name=None, 
              instance_token='ego', 
              sample_token=None,  
              sample_idx=None,
              sim_ego_pos=None,
              sim_ego_quat=None,
              sim_ego_speed=None,
              sim_ego_steering=None
            ):
        if scene_name is None:
            scene_idx = np.random.choice(len(self.scene_list))
            self.scene_name = self.scene_list[scene_idx][:-4]
        else:
            self.scene_name = scene_name
        
        if self.config['data_type'] == 'scene':
            self.scene_data = pd.read_pickle(os.path.join(self.config['data_dir'], self.scene_name+".pkl"))
        else:
            self.scene_data = self.df.query("scene_name == @scene_name")
        
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

        self.update_row(instance_token,self.instance_sample_idx_list[self.df_idx])
        
        if 'current_agent_steering' in self.scene_data.keys(): 
            csteering, psteering, fsteering = calculate_steering(self.r.agent_token,
                                                                current_steering=self.r.current_agent_steering,
                                                                past_steering=self.r.past_agent_steering,
                                                                future_steering=self.r.future_agent_steering,
                                                                current_quat=self.r.current_agent_quat,
                                                                past_quat=self.r.past_agent_quat,
                                                                future_quat=self.r.future_agent_quat)
        else:
            return None            
            
        # currents #
        self.sample_idx = self.instance_sample_idx_list[self.df_idx]
        self.sample_token = self.instance_sample_token_list[self.df_idx]
        self.update_row(self.instance_token, self.sample_idx)
        self.map = NuScenesMap(dataroot=mini_path, map_name=self.r.scene_location)
        
        if sim_ego_pos is None:
            self.sim_ego_pos_gb = self.r.current_agent_pos
        else:
            self.sim_ego_pos_gb = sim_ego_pos
        if sim_ego_quat is None:
            self.sim_ego_quat_gb = self.r.current_agent_quat
        else:
            self.sim_ego_quat_gb = sim_ego_quat
        if sim_ego_speed is None:
            self.sim_ego_speed = self.r.current_agent_speed
        else:
            self.sim_ego_speed = sim_ego_speed
        if sim_ego_steering is None:
            self.sim_ego_yaw_rate = csteering
        else:
            self.sim_ego_yaw_rate = sim_ego_steering
        self.sim_ego_yaw = quaternion_yaw(Quaternion(self.sim_ego_quat_gb))
        
        ''' for q in self.r.future_agent_quat:
            print(quaternion_yaw(Quaternion(q)))
        print(self.r.future_agent_steering)
        print(self.r.future_agent_speed) '''
        
        # histories #
        past_raster = np.array([np.asarray(Image.open(os.path.join(self.config['raster_dir'], str(p)))) for p in self.r.past_agent_raster_path])
        self.sim_ego_raster_image_history = [r for r in past_raster]
        self.sim_ego_goal = self.scene_data.current_agent_pos.tolist()[-1]
        self.sim_ego_pos_history = self.scene_data.iloc[0].past_agent_pos.tolist() + self.scene_data.current_agent_pos.tolist()[:self.df_idx+1]
        self.sim_ego_quat_history = self.scene_data.iloc[0].past_agent_quat.tolist() + self.scene_data.current_agent_quat.tolist()[:self.df_idx+1]
        self.sim_ego_speed_history = self.scene_data.iloc[0].past_agent_speed.tolist() + self.scene_data.current_agent_speed.tolist()[:self.df_idx+1]
        self.sim_ego_steering_history = psteering 
        
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

        fig, ax, other =  render(self.graphics, render_info, self.config)
        if 'traffic_graph' in self.config['render_elements']:
            all_pos = np.vstack([self.r.current_neighbor_pos, self.r.current_agent_pos])
            for p1 in all_pos:
                for p2 in all_pos:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=0.5,alpha=0.3, zorder=400)

        if 'groundtruth' in self.config['render_elements']:
            #####################
            # Plot GroundTruths #
            #####################
            sim_ego_pos = self.all_info['sim_ego_pos_gb']
            #### plot neighbor connection lines ####
            all_pos = np.vstack([self.r.current_neighbor_pos, self.r.current_agent_pos])
            
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
        
        if 'lanes' in self.config['render_elements']:
            #ax.plot(self.all_info['sim_ego_future_lanes'][0][:10,], self.all_info['sim_ego_future_lanes'][0][:10,:], linestyle='-.', color='grey', linewidth=2, zorder=750)            
            ks = list(self.r.keys())
            ''' if 'past_agent_lane' in ks:
                past_agent_lane = self.r['past_agent_lane']
                ax.plot(past_agent_lane[:,0], past_agent_lane[:,1], linestyle='-.', color='grey', linewidth=2, zorder=750)
                
            if 'future_agent_lane' in ks:
                future_agent_lane = self.r['future_agent_lane']
                ax.plot(future_agent_lane[:,0], future_agent_lane[:,1], linestyle='-.', color='green', linewidth=2, zorder=750)
                
            for pl in self.r.past_neighbor_lane:
                ax.plot(pl[:,0], pl[:,1], linestyle='-.', color='grey', linewidth=1, zorder=750)
            
            for fl in self.r.future_neighbor_lane:
                ax.plot(fl[:,0], fl[:,1], linestyle='-.', color='green', linewidth=2, zorder=750) '''
        
        # future agent lane from dataset #
        future_agent_lane_local = self.all_info['adapt_one_row_data']['future_lane']
        future_agent_lane_gb = convert_local_coords_to_global(future_agent_lane_local, self.sim_ego_pos_gb, self.sim_ego_quat_gb)
        ax.plot(future_agent_lane_gb[:,0], future_agent_lane_gb[:,1], linestyle='-.', color='grey', linewidth=2, zorder=750)
            
        return fig, ax, other

    def step(self, action=None, render_info={}, save_img_dir=None):
        if self.py_logger is not None:
            self.py_logger.debug(f"received action: {action}")
        
        #print(action, (self.all_info['ego_speed'], self.all_info['ego_yaw_rate']))
        #### render ####
        fig, ax = None, None
        render_other = {}
        if len(self.config['render_type']) > 0:
            fig, ax, render_other = self.render(render_info, save_img_dir)

        if action is None:
            self.sim_ego_pos_gb = self.all_info['ego_pos_gb']
            self.sim_ego_quat_gb = self.all_info['ego_quat_gb']
            self.sim_ego_speed = self.all_info['ego_speed']
            self.sim_ego_yaw_rate = self.all_info['ego_yaw_rate']


        if self.config['control_mode'] == 'position' and action is not None:
            direction = action - self.sim_ego_pos_gb
            self.sim_ego_speed = np.linalg.norm(direction) / 0.5
            heading = np.arctan2(direction[1], direction[0])
            self.sim_ego_yaw_rate = (heading - self.sim_ego_yaw) / 0.5
            self.sim_ego_yaw = heading
            q = Quaternion(axis=[0,0,1], angle=heading)
            self.sim_ego_quat_gb = np.array([q[0], q[1], q[2], q[3]])
            self.sim_ego_pos_gb = action
            
        #if self.config['control_mode'] == 'trajectory' and action is not None:
        #    self.sim_ego_pos_gb = action[0]
            
        if self.config['control_mode'] == 'kinematics' and action is not None:
            #### using a unicycle model ####
            self.sim_ego_speed = action[0]
            self.sim_ego_yaw_rate = action[1]

            dx_local = self.sim_ego_speed * np.array([np.cos(self.sim_ego_yaw), np.sin(self.sim_ego_yaw)]) * 0.5

            self.sim_ego_pos_gb += dx_local

            self.sim_ego_yaw += self.sim_ego_yaw_rate * 0.5
            q = Quaternion(axis=[0,0,1], degrees=np.rad2deg(self.sim_ego_yaw))
            self.sim_ego_quat_gb = np.array([q[0], q[1], q[2], q[3]])


        self.df_idx += 1
        self.update_all_info()
        done = False
        if self.df_idx >= len(self.instance_sample_idx_list):
            done = True

        other = {
            'render_fig': fig,
            'render_ax': ax
        }
        other.update(render_other)

        
        return self.get_observation(), done, other

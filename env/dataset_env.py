import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from utils.utils import convert_local_coords_to_global, convert_global_coords_to_local
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


class NuScenesDatasetEnv(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = {
            'NuScenesAgent_config':{},
            'data_path': None,
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

        #### Initialize ####
        self.all_info = {}
        self.reset()

        #### load dataset ####
        self.dataset = pd.read_pickle(self.config['data_dir'])
        
    def update_all_info(self):
        pass

    def reset(scene_name=None, instance_token=None):
        if scene_name is None:
            pass

    def get_observation(self):
        return self.all_info


    def render(self, render_info={}):
        render_info['sim_ego_quat_gb'] = self.sim_ego_quat_gb
        render_info['sim_ego_pos_gb'] = self.sim_ego_pos_gb
        render_info['ap_speed'] = self.ap_speed
        render_info['ap_steering'] = self.ap_steering
        render_info['ap_timesteps'] = self.ap_timesteps
        render_info['scene_name'] = self.scene['name']
        render_info['all_info'] = self.all_info
        render_info['sample_token'] = self.sample['token']
        render_info['intance_token'] = self.instance_token
        render_info['sample_idx'] = self.sample_idx

        return render(self.graphics, render_info, self.config)

    def step(self, action=None, render_info={}):
        if self.py_logger is not None:
            self.py_logger.debug(f"received action: {action}")

        #### render ####
        fig, ax = None, None
        if len(self.config['render_type']) > 0:
            fig, ax = self.render(render_info)

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


        self.sample_idx += 1
        self.update_all_info()
        done = False
        other = {
            'render_ax': ax
        }
        return self.get_observation(), done, other

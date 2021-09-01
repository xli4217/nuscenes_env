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
from .env_render import render

class NuScenesEnv(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = {
            'NuScenesAgent_config':{},
            'Sensor_config': {},
            'SceneGraphics_config': {},
            'render_paper_ready': False,
            'render_type': [],
            'render_elements': ['sim_ego'], # can contain ['sensor_info', 'sim_ego', 'human_ego', 'control_plots']
            'patch_margin': 30,
            'save_image_dir': None,
            'scenes_range': [0,10],
            # contains ['center lane', 'raster_image']
            'all_info_fields': [], 
            'observations': [],
            'control_mode': 'position' # can be 'position' or 'kinematics'
        }
        self.config.update(config)

        super().__init__(config=self.config['NuScenesAgent_config'], helper=helper, py_logger=py_logger, tb_logger=tb_logger)
        
        
        #### Instantiate Sensor ####
        sensor_config = copy.deepcopy(self.config['Sensor_config'])
        sensor_config['NuScenesAgent_config'] = self.config['NuScenesAgent_config']
        sensor_config['load_dataset'] = False
        self.sensor = Sensor(sensor_config, self.helper, self.py_logger, self.tb_logger)

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
        self.py_logger = py_logger
        self.all_info = {}
        self.reset()
        
    def update_all_info(self):

        #### scene info ####
        scene_info = {
            'scene_token': self.scene['token'],
            'scene_description': self.scene['description'],
            'scene_name': self.scene['name'],
            'scene_nbr_samples': self.scene['nbr_samples']
        }
        self.all_info['scene_info'] = scene_info

        #### sensor info ####
        if self.instance_token is None:
            instance_based = False
        else:
            instance_based = True
        sensor_info = self.sensor.get_info(self.sample['token'], ego_pos=self.sim_ego_pos_gb, ego_quat=self.sim_ego_quat_gb, instance_based=instance_based)

        filtered_agent_info = self.filter_agent_info(sensor_info['agent_info'])
        sensor_info['agent_info'] = filtered_agent_info
        self.all_info['sensor_info'] = sensor_info


        #### sample info ####
        self.all_info['sample_idx'] = self.sample_idx
        self.all_info['sample_token'] = self.sample['token']
        self.all_info['time'] = self.time

        if self.instance_token is None:
            sample_data = self.helper.data.get('sample_data', self.sample['data']['CAM_FRONT'])
            ego_pose = self.helper.data.get('ego_pose', sample_data['ego_pose_token'])
        else:
            ego_pose = {
                'translation': self.inst_ann['translation'],
                'rotation': self.inst_ann['rotation']
            }

        #### ego pose ####
        ego_yaw = Quaternion(ego_pose['rotation'])
        self.ego_yaw = quaternion_yaw(ego_yaw)
        #self.ego_yaw = angle_of_rotation(ego_yaw)

        self.all_info['ego_init_pos_gb'] = np.array(self.init_ego_pos)
        self.all_info['ego_init_quat_gb'] = np.array(self.init_ego_quat)
        self.all_info['ego_pos_gb'] = np.array(ego_pose['translation'])[:2]
        self.all_info['ego_quat_gb'] = np.array(ego_pose['rotation'])
        self.all_info['ego_yaw_rad'] = self.ego_yaw

        idx = min(self.sample_idx, len(sensor_info['can_info']['ego_speed_traj'])-1)
        ego_speed = sensor_info['can_info']['ego_speed_traj'][idx]
        idx = min(self.sample_idx, len(sensor_info['can_info']['ego_rotation_rate_traj'])-1)
        ego_yaw_rate = sensor_info['can_info']['ego_rotation_rate_traj'][idx][-1]
        idx_plus_one = min(self.sample_idx+1, len(sensor_info['can_info']['ego_rotation_rate_traj'])-1)
        self.next_ego_yaw_rate = sensor_info['can_info']['ego_rotation_rate_traj'][idx_plus_one][-1]
        
        self.all_info['ego_speed'] = ego_speed
        self.all_info['ego_yaw_rate'] = ego_yaw_rate

        self.all_info['ego_pos_traj'] = np.array(self.true_ego_pos_traj)
        self.all_info['ego_quat_traj'] = np.array(self.true_ego_quat_traj)
        self.all_info['ego_past_pos'] = np.array(self.true_ego_pos_traj)[0:self.sample_idx]
        self.all_info['ego_future_pos'] = np.array(self.true_ego_pos_traj)[self.sample_idx:]
        self.all_info['ego_past_quat'] = np.array(self.true_ego_quat_traj)[0:self.sample_idx]
        self.all_info['ego_future_quat'] = np.array(self.true_ego_quat_traj)[self.sample_idx:]


        if self.sim_ego_pos_gb is None:
            self.sim_ego_pos_gb = np.array(ego_pose['translation'])[:2]
            self.sim_ego_quat_gb = np.array(ego_pose['rotation'])
            self.sim_ego_speed = self.all_info['ego_speed']
            self.sim_ego_yaw_rate = self.all_info['ego_yaw_rate']

            sim_ego_yaw = Quaternion(self.sim_ego_quat_gb)
            self.sim_ego_yaw = quaternion_yaw(sim_ego_yaw)
            #self.sim_ego_yaw = angle_of_rotation(sim_ego_yaw)

        #### sim ego pose ####
        self.sim_ego_pos_traj.append(self.sim_ego_pos_gb.copy())
        self.sim_ego_quat_traj.append(self.sim_ego_quat_gb.copy())

        self.all_info['sim_ego_pos_gb'] = self.sim_ego_pos_gb
        self.all_info['sim_ego_quat_gb'] = self.sim_ego_quat_gb
        self.all_info['sim_ego_yaw_rad'] = self.sim_ego_yaw
        self.all_info['sim_ego_speed'] = self.sim_ego_speed
        self.all_info['sim_ego_yaw_rate'] = self.sim_ego_yaw_rate
        self.all_info['sim_ego_pos_traj'] = self.sim_ego_pos_traj
        self.all_info['sim_ego_quat_traj'] = self.sim_ego_quat_traj

        #### future lanes ####
        self.all_info['gt_future_lanes'] = get_future_lanes(self.nusc_map, self.all_info['ego_pos_gb'], self.all_info['ego_quat_gb'], frame='global')
        
        self.all_info['future_lanes'] = get_future_lanes(self.nusc_map, self.sim_ego_pos_gb, self.sim_ego_quat_gb, frame='global')

        #### neighbor pos ####
        current_sim_neighbor_pos = []
        for agent in sensor_info['agent_info']:
            current_sim_neighbor_pos.append(agent['translation'])

        self.all_info['current_sim_neighbor_pos'] = np.array(current_sim_neighbor_pos)
            
        #### rasterized image ####
        if 'raster_image' in self.config['all_info_fields'] and self.rasterizer is not None:
            #### ego raster img ####
            ego_raster_img = self.rasterizer.make_input_representation(instance_token=None, sample_token=self.sample_token, ego=True, ego_pose=ego_pose, include_history=False)
            #ego_raster_img = np.transpose(ego_raster_img, (2,0,1))
            self.all_info['raster_image'] = ego_raster_img

            #### sim ego raster img ####
            sim_ego_pose = {
                'translation': self.sim_ego_pos_gb,
                'rotation': self.sim_ego_quat_gb
            }

            sim_ego_raster_img = self.rasterizer.make_input_representation(instance_token=None, sample_token=self.sample_token, ego=True, ego_pose=sim_ego_pose, include_history=False)
            #sim_ego_raster_img = np.transpose(sim_ego_raster_img, (2,0,1))
            self.all_info['sim_ego_raster_image'] = sim_ego_raster_img
        else:
            self.all_info['raster_image'] = None
            self.all_info['sim_ego_raster_image'] = None
            
    def reset_ego(self, scene_name=None, scene_idx=None, sample_idx=0):
        if scene_name is None and scene_idx is None:
            scene_list = np.arange(self.config['scenes_range'][0], self.config['scenes_range'][1])
            scene_idx = np.random.choice(scene_list)
            self.scene = self.nusc.scene[scene_idx]
        elif scene_name is None and scene_idx is not None:
            self.scene = self.nusc.scene[scene_idx]
        elif scene_name is not None and scene_idx is None:
            for scene in self.nusc.scene:
                if scene['name'] == scene_name:
                    self.scene = scene
                    break

        else:
            raise ValueError('can not have both scene_name and scene_idx as input')

        print(f"current scene: {self.scene['name']}")
        self.scene_log = self.helper.data.get('log', self.scene['log_token'])
        self.nusc_map = self.nusc_map_dict[self.scene_log['location']]

        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)
        self.sample_idx = sample_idx
        s_idx = 0
        while self.sample['next'] != "" and s_idx != self.sample_idx:
            self.sample = self.nusc.get('sample', self.sample['next'])
            s_idx += 1
        self.sample_token = self.sample['token']

        #### get ego traj ####
        #sample_tokens = self.nusc.field2token('sample', 'scene_token', self.scene['token'])
        sample_tokens = []
        sample = copy.deepcopy(self.sample)
        while sample['next'] != "":
            sample_tokens.append(sample['token'])
            sample = self.nusc.get('sample', sample['next'])

        self.true_ego_pos_traj = []
        self.true_ego_quat_traj = []
        for sample_token in sample_tokens:
            sample_record = self.nusc.get('sample', sample_token)

            # Poses are associated with the sample_data. Here we use the lidar sample_data.
            sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])

            self.true_ego_pos_traj.append(pose_record['translation'])
            self.true_ego_quat_traj.append(pose_record['rotation'])

        self.init_ego_pos = self.true_ego_pos_traj[0]
        self.init_ego_quat = self.true_ego_quat_traj[0]
        self.sim_ego_pos_traj = []
        self.sim_ego_quat_traj = []
        self.center_agent = 'ego'

    def reset_ado(self, instance_token):
        self.instance_token = instance_token
        for inst in self.nusc.instance:
            if inst['token'] == self.instance_token:
                self.nbr_ann = inst['nbr_annotations']
                self.first_ann_token = inst['first_annotation_token']
                self.last_ann_token = inst['last_annotation_token']
                self.ann_token = self.first_ann_token
                self.inst_ann = self.nusc.get('sample_annotation', self.ann_token)

                self.sample_token = self.inst_ann['sample_token']
                self.sample = self.nusc.get('sample', self.sample_token)
                self.scene_token = self.sample['scene_token']
                self.scene = self.nusc.get('scene', self.scene_token)
                self.scene_log = self.helper.data.get('log', self.scene['log_token'])
                self.nusc_map = self.nusc_map_dict[self.scene_log['location']]
                self.init_ego_pos = self.inst_ann['translation']
                self.init_ego_quat = self.inst_ann['rotation']
                break

        self.true_ego_pos_traj = []
        self.true_ego_quat_traj = []

        agent_futures = self.helper.get_future_for_agent(self.instance_token, self.sample_token, 20, in_agent_frame=True, just_xy=False)
        for a in agent_futures:
            self.true_ego_pos_traj.append(a['translation'])
            self.true_ego_quat_traj.append(a['rotation'])

        self.center_agent = 'ado'

    def reset(self, scene_name=None, scene_idx=None, sample_idx=0, instance_token=None):
        if instance_token == 'ego':
            instance_token = None    
        
        self.sim_ego_pos_gb = None
        self.sim_ego_quat_gb = None
        self.sample_idx = 0
        self.sample_token = None
        self.instance_token = None
        self.inst_ann = None
        self.center_agent = None
        self.time = 0            
            
        if 'control_plots' in self.config['render_elements']:
            if self.config['control_mode'] != 'kinematics':
                raise ValueError('action plots need to be generated in kinematics control mode')
            self.ap_timesteps = collections.deque(maxlen=10)
            self.ap_speed = collections.deque(maxlen=10)
            self.ap_steering = collections.deque(maxlen=10)

        if instance_token is None:
            self.reset_ego(scene_name, scene_idx, sample_idx)
        else:
            self.reset_ado(instance_token)

        self.update_all_info()
        return self.get_observation()

    def get_observation(self):
        return self.all_info

    def filter_agent_info(self, agent_info):
        filtered_agent_info = []

        for agent in agent_info:
            #if ('vehicle' in agent['category'] and 'parked' not in agent['attribute']) or ('pedestrian' in agent['category'] and 'stroller' not in agent['category'] and 'wheelchair' not in agent['category']):
            if ('vehicle' in agent['category'] and 'parked' not in agent['attribute']):
                filtered_agent_info.append(agent)

        return filtered_agent_info

    def render(self, render_info={}, save_img_dir=None):
        render_info['sim_ego_quat_gb'] = self.sim_ego_quat_gb
        render_info['sim_ego_pos_gb'] = self.sim_ego_pos_gb
        render_info['ap_speed'] = None #self.ap_speed
        render_info['ap_steering'] = None #self.ap_steering
        render_info['ap_timesteps'] = None #self.ap_timesteps
        render_info['scene_name'] = self.scene['name']
        render_info['all_info'] = self.all_info
        render_info['sample_token'] = self.sample['token']
        render_info['instance_token'] = self.instance_token
        render_info['sample_idx'] = self.sample_idx
        render_info['save_image_dir'] = save_img_dir

        return render(self.graphics, render_info, self.config)
        
    def step(self, action:np.ndarray=None, render_info={}, save_img_dir=None):
        if self.py_logger is not None:
            self.py_logger.debug(f"received action: {action}")
        #### render ####
        fig, ax = None, None
        if len(self.config['render_type']) > 0:
            if 'control_plots' in self.config['render_elements'] and action is not None:
                self.ap_speed.append(action[0])
                self.ap_steering.append(action[1])
                self.ap_timesteps.append(self.time)
            fig, ax = self.render(render_info, save_img_dir)

        if action is None:
            self.sim_ego_pos_gb = self.all_info['ego_pos_gb']
            self.sim_ego_quat_gb = self.all_info['ego_quat_gb']
            self.sim_ego_speed = self.all_info['ego_speed']
            self.sim_ego_yaw_rate = self.all_info['ego_yaw_rate']


        if self.config['control_mode'] == 'position' and action is not None:
            self.sim_ego_pos_gb = action
            self.sim_ego_speed = np.linalg.norm(action - self.sim_ego_pos_gb)/0.5
            
            ### TODO: fix sim ego yaw rate update
            self.sim_ego_yaw_rate = self.next_ego_yaw_rate
            
            ### TODO: fix this quat update ####
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

        done = False
        if self.inst_ann is None:
            if self.sample['next'] == "":
                done = True
            if not done:
                self.sample = self.nusc.get('sample', self.sample['next'])
                self.sample_token = self.sample['token']
        else:
            if self.inst_ann['next'] == "":
                done = True
            if not done:
                self.ann_token = self.inst_ann['next']
                self.inst_ann = self.nusc.get('sample_annotation', self.ann_token)
                self.sample_token = self.inst_ann['sample_token']
                self.sample = self.nusc.get('sample', self.sample_token)

        self.sample_idx += 1
        self.time += 0.5
        self.update_all_info()
        other = {
            'render_ax': ax
        }
        return self.get_observation(), done, other



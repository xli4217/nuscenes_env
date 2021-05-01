import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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

from input_representation.static_layers import StaticLayerRasterizer
from input_representation.agents import AgentBoxesWithFadedHistory
from input_representation.interface import InputRepresentation
from input_representation.combinators import Rasterizer

from pathlib import Path
from .env_utils import *

class NuScenesEnv(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = {
            'NuScenesAgent_config':{},
            'Sensor_config': {},
            'SceneGraphics_config': {},
            'render_paper_ready': False,
            'render_type': [],
            'render_elements': ['sim_ego'], # can contain ['sensor_info', 'sim_ego']
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
        self.all_info = {}
        self.reset()

    def update_all_info(self):

        #### scene info ####
        scene_info = {
            'scene_token': self.scene['token'],
            'scene_description': self.scene['description'],
            'scene_name': self.scene['name'],
        }
        self.all_info['scene_info'] = scene_info

        #### sample info ####
        self.all_info['sample_idx'] = self.sample_idx
        self.all_info['sample_token'] = self.sample['token']
        
        sample_data = self.helper.data.get('sample_data', self.sample['data']['CAM_FRONT'])
        ego_pose = self.helper.data.get('ego_pose', sample_data['ego_pose_token'])

        #### ego pose ####
        ego_yaw = Quaternion(ego_pose['rotation'])
        self.ego_yaw = quaternion_yaw(ego_yaw)
        #self.ego_yaw = angle_of_rotation(ego_yaw)

        self.all_info['ego_init_pos_gb'] = np.array(self.init_ego_pos)
        self.all_info['ego_init_quat_gb'] = np.array(self.init_ego_quat)
        self.all_info['ego_pos_gb'] = np.array(ego_pose['translation'])[:2]
        self.all_info['ego_quat_gb'] = np.array(ego_pose['rotation'])
        self.all_info['ego_yaw_rad'] = self.ego_yaw
        
        self.all_info['ego_past_pos'] = np.array(self.true_ego_pos_traj)[0:self.sample_idx]
        self.all_info['ego_future_pos'] = np.array(self.true_ego_pos_traj)[self.sample_idx:]
        self.all_info['ego_past_quat'] = np.array(self.true_ego_quat_traj)[0:self.sample_idx]
        self.all_info['ego_future_quat'] = np.array(self.true_ego_quat_traj)[self.sample_idx:]

        
        if self.sim_ego_pos_gb is None:
            self.sim_ego_pos_gb = np.array(ego_pose['translation'])[:2]
            self.sim_ego_quat_gb = np.array(ego_pose['rotation'])

        sim_ego_yaw = Quaternion(self.sim_ego_quat_gb)
        self.sim_ego_yaw = quaternion_yaw(sim_ego_yaw)
        #self.sim_ego_yaw = angle_of_rotation(sim_ego_yaw)

        #### sim ego pose ####
        self.all_info['sim_ego_pos_gb'] = self.sim_ego_pos_gb
        self.all_info['sim_ego_quat_gb'] = self.sim_ego_quat_gb
        self.all_info['sim_ego_yaw_rad'] = self.sim_ego_yaw
        
        #### future lanes ####
        self.all_info['future_lanes'] = get_future_lanes(self.nusc_map, self.sim_ego_pos_gb, self.sim_ego_quat_gb, frame='global')

        #### sensor info ####
        sensor_info = self.sensor.get_info(self.sample['token'], ego_pos=self.sim_ego_pos_gb, ego_quat=self.sim_ego_quat_gb)

        filtered_agent_info = self.filter_agent_info(sensor_info['agent_info'])
        sensor_info['agent_info'] = filtered_agent_info
        self.all_info['sensor_info'] = sensor_info

        #### rasterized image (TODO: support for sim ego and ado) ####
        if 'raster_image' in self.config['all_info_fields'] and self.rasterizer is not None:
            #### ego raster img ####
            ego_raster_img = self.rasterizer.make_input_representation(instance_token=None, sample_token=self.sample_token, ego=True, ego_pose=ego_pose)
            ego_raster_img = np.transpose(ego_raster_img, (2,0,1))
            self.all_info['raster_image'] = ego_raster_img

            #### sim ego raster img ####
            sim_ego_pose = {
                'translation': self.sim_ego_pos_gb,
                'rotation': self.sim_ego_quat_gb
            }
            sim_ego_raster_img = self.rasterizer.make_input_representation(instance_token=None, sample_token=self.sample_token, ego=True, ego_pose=sim_ego_pose)
            sim_ego_raster_img = np.transpose(sim_ego_raster_img, (2,0,1))
            self.all_info['sim_ego_raster_image'] = sim_ego_raster_img

            
    def reset(self, scene_name=None, scene_idx=None):
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

        self.sim_ego_pos_gb = None
        self.sim_ego_quat_gb = None

        self.sample_idx = 0

        #### get ego traj ####
        sample_tokens = self.nusc.field2token('sample', 'scene_token', self.scene['token'])
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
        
    def render(self, render_info={}):
        if 'image' in self.config['render_type']:
            sim_ego_yaw = Quaternion(self.sim_ego_quat_gb)
            sim_ego_yaw = quaternion_yaw(sim_ego_yaw)
            sim_ego_yaw = angle_of_rotation(sim_ego_yaw)
            sim_ego_yaw = np.rad2deg(sim_ego_yaw)

            ego_traj = None
            if 'sim_ego' in self.config['render_elements']:
                ego_traj = {
                    # 'lane': {
                    #     'traj': self.all_info['future_lanes'][0],
                    #     'color': 'green'
                    # },
                    'sim_ego':{
                        'pos': self.sim_ego_pos_gb,
                        'yaw': sim_ego_yaw,
                        'traj': np.zeros((4,2)),
                        'color': 'yellow'
                    }
                }

            if self.config['save_image_dir'] is not None:
                save_img_dir = os.path.join(self.config['save_image_dir'], str(self.scene['name']))
                if not os.path.exists(save_img_dir):
                    os.makedirs(save_img_dir, exist_ok=True)

            sensor_info = None
            if 'sensor_info' in self.config['render_elements']:
                sensor_info = self.all_info['sensor_info']

            ado_traj_dict = None
            if 'ado_traj_dict' in render_info.keys():
                ado_traj_dict = render_info['ado_traj_dict']

            costmap_contour = None
            if 'costmap_contour' in render_info.keys():
                costmap_contour = render_info['costmap_contour']

            other_images_to_be_saved = None
            if 'sim_ego_raster_image' in self.all_info.keys():
                other_images_to_be_saved = {
                    'raster': np.transpose(self.all_info['sim_ego_raster_image'], (1,2,0))
                }

            render_additional = None
            if 'lines' in render_info.keys():
                render_additional['lines'] = render_info['lines']
            if 'scatters' in render_info.keys():
                render_additional['scatters'] = render_info['scatters']
                
            fig, ax = self.graphics.plot_ego_scene(sample_token=self.sample['token'],
                                                   ego_traj=ego_traj,
                                                   ado_traj=ado_traj_dict,
                                                   contour=costmap_contour,
                                                   save_img_dir=save_img_dir,
                                                   idx=str(self.sample_idx).zfill(2),
                                                   sensor_info=sensor_info,
                                                   paper_ready=self.config['render_paper_ready'],
                                                   other_images_to_be_saved=other_images_to_be_saved,
                                                   render_additional = render_additional
            )
            plt.show()
            
    def step(self, action:np.ndarray=None, render_info={}):
        if len(self.config['render_type']) > 0:
            self.render(render_info)
        
        if self.config['control_mode'] == 'position':
            self.sim_ego_pos_gb = action
            self.sim_ego_quat_gb = self.all_info['ego_quat_gb']

        if self.config['control_mode'] == 'kinematics':
            #### using a unicycle model ####
            sim_ego_speed = action[0]
            sim_ego_yaw_rate = action[1]

            dx_local = sim_ego_speed * np.array([np.cos(self.sim_ego_yaw), np.sin(self.sim_ego_yaw)]) * 0.5

            self.sim_ego_pos_gb += dx_local

            self.sim_ego_yaw += sim_ego_yaw_rate * 0.5
            q = Quaternion(axis=[0,0,1], degrees=np.rad2deg(self.sim_ego_yaw))
            self.sim_ego_quat_gb = [q[0], q[1], q[2], q[3]]
            
        done = False
        if self.sample['next'] == "":
            done = True

        if not done:
            self.sample = self.nusc.get('sample', self.sample['next'])

        self.update_all_info()
        self.sample_idx += 1
        return self.get_observation(), done
        

    def make_video_from_images(self, image_dir:str=None, video_save_dir:str=None):
        img_fn_list = [str(p).split('/')[-1] for p in Path(image_dir).rglob('*.png')]
        birdseye_img_list = [p for p in img_fn_list if 'birdseye' in p and 'checkpoint' not in p]
        birdseye_idx = np.argsort(np.array([int(p[:2]) for p in birdseye_img_list]))
        birdseye_img_list = np.array(birdseye_img_list)[birdseye_idx]

        camera_img_list =  [p for p in img_fn_list if 'camera' in p and 'checkpoint' not in p]
        camera_idx = np.argsort(np.array([int(p[:2]) for p in camera_img_list]))
        camera_img_list = np.array(camera_img_list)[camera_idx]

        # raster_img_list =  [p for p in img_fn_list if 'raster' in p and 'checkpoint' not in p]
        # raster_idx = np.argsort(np.array([int(p[:2]) for p in raster_img_list]))
        # raster_img_list = np.array(raster_img_list)[raster_idx]
 
 
        fig = plt.figure(figsize=(15, 15), constrained_layout=False)
        gs = fig.add_gridspec(nrows=6, ncols=6, wspace=0.01)
        ax1 = fig.add_subplot(gs[:4, :6]) # bird-view
        ax2 = fig.add_subplot(gs[4:, :4]) # front camera
        #ax3 = fig.add_subplot(gs[4:, 4:6]) # raster
        ax1.axis('off')
        ax2.axis('off')
        #ax3.axis('off')

        i = 0
        camera = Camera(fig)
        #for p_birdseye, p_camera, p_raster in tqdm.tqdm(zip(birdseye_img_list, camera_img_list, raster_img_list)):
        for p_birdseye, p_camera in tqdm.tqdm(zip(birdseye_img_list, camera_img_list)):
            # if i > 2:
            #     break
            birdseye_img = plt.imread(os.path.join(image_dir, p_birdseye))
            camera_img = plt.imread(os.path.join(image_dir, p_camera))
            #raster_img = plt.imread(os.path.join(image_dir, p_raster))
            ax1.imshow(birdseye_img)
            ax2.imshow(camera_img)
            #ax3.imshow(raster_img)
            camera.snap()

        animation = camera.animate()

        if video_save_dir is not None:
            animation.save(video_save_dir+'/video.mp4')
        return animation

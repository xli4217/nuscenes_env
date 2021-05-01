import json
from future.utils import viewitems
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import cv2
import io
import sklearn
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
import os
import tqdm
from scipy.ndimage import rotate
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
import shapely

from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global, convert_global_coords_to_local
import descartes

from graphics.nuscenes_agent import NuScenesAgent
from utils.utils import transform_mesh2D, translate_mesh2D, rotate_mesh2D


class Sensor(NuScenesAgent):
    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):
        self.config =  {
            'NuScenesAgent_config':{}
        }
        self.config.update(config)

        super().__init__(config=self.config['NuScenesAgent_config'], helper=helper, py_logger=py_logger, tb_logger=tb_logger)



    def update_all_info(self):
        pass
            
    def get_info(self, sample_token: str, ego_pos:np.ndarray=None, ego_quat:np.ndarray=None, instance_based=False):

        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        scene_log = self.nusc.get('log', scene['log_token'])
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=scene_log['location'])
        sample_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])

        if ego_pos is None:
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            ego_pos = np.array(ego_pose['translation'])
        if ego_quat is None:
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            ego_quat = np.array(ego_pose['rotation'])

        ego_yaw = Quaternion(ego_quat)
        ego_yaw = quaternion_yaw(ego_yaw)
        ego_yaw_rad = angle_of_rotation(ego_yaw)
        ego_yaw_degrees = np.rad2deg(ego_yaw_rad)

        
        ego_info = {
            'translation': ego_pos,
            'rotation_deg': ego_yaw_degrees,
            'rotation_quat': ego_quat,
            'velocity': 0,
            'acceleration': 0,
            'heading_change_rate':0
        }

        #### define patch ####
        sensing_patch_width = 60
        sensing_patch_length = 60

        # patch_center_before_rotation = np.array([ego_pos[0],
        #                                          ego_pos[1] + sensing_patch_length/2])

        # lower_left_before_rotation = np.array([ego_pos[0] - sensing_patch_width/2,
        #                                        ego_pos[1] - sensing_patch_length/2])

        # sensing_patch = self.get_patch_coord(patch_box=(patch_center_before_rotation[0],
        #                                                 patch_center_before_rotation[1],
        #                                                 sensing_patch_length,
        #                                                 sensing_patch_width),
        #                                      rotate_center=(ego_pos[0], ego_pos[1]),
        #                                      patch_angle=-ego_yaw_degrees)

        
        patch_center_before_rotation = np.array([ego_pos[0],
                                                 ego_pos[1]])

        lower_left_before_rotation = np.array([ego_pos[0] - sensing_patch_width/2,
                                               ego_pos[1] - sensing_patch_length/2])

        sensing_patch_coord_before_rotation = [
            ego_pos[0] - sensing_patch_width/2,
            ego_pos[1] - sensing_patch_length/2,
            ego_pos[0] + sensing_patch_width/2,
            ego_pos[1] + sensing_patch_length/2 
        ]

        ## generate sensing patch mesh
        x = np.arange(sensing_patch_coord_before_rotation[0], sensing_patch_coord_before_rotation[2], 0.2)
        y = np.arange(sensing_patch_coord_before_rotation[1], sensing_patch_coord_before_rotation[3], 0.2)

        X, Y = np.meshgrid(x, y)
        ### apply rotation
        X, Y = rotate_mesh2D(pos=ego_pos, rot_rad=ego_yaw_rad, X=X, Y=Y, frame='current')
        
        ## generate sensing patch shapely polygon
        sensing_patch = self.get_patch_coord(patch_box=(patch_center_before_rotation[0],
                                                        patch_center_before_rotation[1],
                                                        sensing_patch_length,
                                                        sensing_patch_width),
                                             rotate_center=(ego_pos[0], ego_pos[1]),
                                             patch_angle=-ego_yaw_degrees)


        sensing_patch_info = {
            'mesh': [X, Y],
            'polygon': sensing_patch
        }
        # get agent_info in patch
        agent_info = self.get_agent_info(sample, sensing_patch)

        # get map info in patch
        map_info = self.get_map_info(ego_pos,
                                     ego_yaw_degrees,
                                     nusc_map,
                                     sensing_patch)

        # get canbus info
        can_info = None
        if not instance_based:
            can_info = self.get_can_info(scene['name'])
        
        sensing_info = {
            'sample_token':sample_token,
            'sensing_patch':sensing_patch_info,
            'ego_info': ego_info,
            'agent_info':agent_info,
            'map_info': map_info,
            'can_info': can_info
        }
        
        return sensing_info

    def get_can_info(self, scene_name:str=None):
        pose_msg = self.nusc_can.get_messages(scene_name, 'pose')
        ego_accel_traj = [pm['accel'] for pm in pose_msg][::25]
        ego_quat_traj = [pm['orientation'] for pm in pose_msg][::25]
        ego_pos_traj = [pm['pos'] for pm in pose_msg][::25]
        ego_rotation_rate_traj = [pm['rotation_rate'] for pm in pose_msg][::25]
        ego_speed_traj = [pm['vel'] for pm in pose_msg][::25]
    
        # Vehicle Monitor message information (original message 2hz, no need to down sample)
        veh_monitor = self.nusc_can.get_messages(scene_name, 'vehicle_monitor')
        ego_speed_traj = [vm['vehicle_speed']*0.28 for vm in veh_monitor]
        ego_steering_deg_traj = [vm['steering'] for vm in veh_monitor]

        # get high level ego motion
        steer = np.array(ego_steering_deg_traj)
        argmax_steer = np.argmax(np.abs(steer))
        max_abs_steer = np.abs(steer)[argmax_steer]
        max_steer_sgn = np.sign(steer[argmax_steer])
        if max_abs_steer > 50:
            if max_steer_sgn > 0:
                ego_high_level_motion = 'LeftTurn'
            else:
                ego_high_level_motion = 'RightTurn'
        else:
            ego_high_level_motion = 'Straight'

        return {
            'ego_accel_traj': ego_accel_traj,
            'ego_quat_traj':ego_quat_traj,
            'ego_pos_traj':ego_pos_traj,
            'ego_rotation_rate_traj':ego_rotation_rate_traj,
            'ego_speed_traj':ego_speed_traj,
            'ego_steering_deg_traj':ego_steering_deg_traj,
            'ego_high_level_motion': ego_high_level_motion
        }
        
    def get_agent_info(self, sample, sensing_patch):
        agent_info = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']
            category = ann['category_name']
            if len(ann['attribute_tokens']) != 0:
                attribute = self.nusc.get('attribute', ann['attribute_tokens'][0])['name']
            else:
                attribute = ""

            agent_pos = [ann['translation'][0], ann['translation'][1]]
            
            include = False
            #### Cars ####
            if 'vehicle' in category and 'parked' not in attribute and self.in_shapely_polygon(agent_pos, sensing_patch):
                include = True
                agent_type = 'car'
                
            #### Plot pedestrians ####
            if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category and self.in_shapely_polygon(agent_pos, sensing_patch):
                include = True
                agent_type = "pedestrian"
                
            if include:
                agent_yaw = Quaternion(ann['rotation'])
                agent_yaw = quaternion_yaw(agent_yaw)
                agent_yaw = angle_of_rotation(agent_yaw)
                agent_yaw_deg = np.rad2deg(agent_yaw)
                
                agent_vel = self.helper.get_velocity_for_agent(instance_token, sample['token'])
                agent_acc = self.helper.get_acceleration_for_agent(instance_token, sample['token'])
                agent_heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample['token'])

                agent_future = self.helper.get_future_for_agent(instance_token,
                                                                sample['token'],
                                                                self.na_config['pred_horizon'],
                                                                in_agent_frame=False,
                                                                just_xy=True)
            
                agent_past = self.helper.get_past_for_agent(instance_token,
                                                            sample['token'],
                                                            self.na_config['obs_horizon'],
                                                            in_agent_frame=False,
                                                            just_xy=True)
   
    
                tmp_agent_info = {
                    'instance_token': instance_token,
                    'type': agent_type,
                    'category': category,
                    'attribute': attribute,
                    'translation': agent_pos,
                    'rotation_deg': agent_yaw_deg,
                    'rotation_quat': ann['rotation'],
                    'velocity': agent_vel,
                    'acceleration': agent_acc,
                    'heading_change_rate': agent_heading_change_rate,
                    'past': agent_past,
                    'future': agent_future
                }

                agent_info.append(tmp_agent_info)
                
        return agent_info
        
    def get_map_info(self,
                     ego_pos,
                     ego_yaw_deg,
                     nusc_map: NuScenesMap,
                     sensing_patch):

        sb = sensing_patch.bounds
        sensing_patch_center = [(sb[0]+sb[2])/2, (sb[1]+sb[3])/2]
        sensing_patch_length = sb[3] - sb[1]
        sensing_patch_width = sb[2] - sb[0]
        
        map_info = {}
                
        #### Get layer information ####
        stop_line_record_list = nusc_map.get_layer_record_in_patch(patch_box=(sensing_patch_center[0],
                                                                    sensing_patch_center[1],
                                                                    sensing_patch_length,
                                                                    sensing_patch_width),
                                                                   patch_angle=-ego_yaw_deg,
                                                                   layer_name='stop_line',
                                                                   rot_center=(ego_pos[0], ego_pos[1]))

        stop_line_data = []
        for r in stop_line_record_list:
            bd = nusc_map.get_bounds('stop_line', r['token'])

            stop_line_data.append({'record': r, 'bounding_box': bd})
            
        map_info['stop_line'] = stop_line_data

        #### Get center lane information ####
        closest_lane_id = nusc_map.get_closest_lane(ego_pos[0], ego_pos[1], radius=2)
        closest_lane_record = nusc_map.get_lane(closest_lane_id)

        closest_lane_poses = np.array(arcline_path_utils.discretize_lane(closest_lane_record, resolution_meters=1))
        map_info['closest_lane'] = {'record': closest_lane_record, 'poses': closest_lane_poses}
        
        incoming_lane_ids = nusc_map.get_incoming_lane_ids(closest_lane_id)
        incoming_lane_data = []
        for incoming_lane_id in incoming_lane_ids:
            record = nusc_map.get_lane(incoming_lane_id)
            poses = np.array(arcline_path_utils.discretize_lane(record, resolution_meters=1))
            incoming_lane_data.append({'record': record, 'poses': poses})

        map_info['incoming_lanes'] = incoming_lane_data
            
        outgoing_lane_ids = nusc_map.get_outgoing_lane_ids(closest_lane_id)
        outgoing_lane_data = []
        for outgoing_lane_id in outgoing_lane_ids:
            record = nusc_map.get_lane(outgoing_lane_id)
            poses = np.array(arcline_path_utils.discretize_lane(record, resolution_meters=1))
            outgoing_lane_data.append({'record': record, 'poses': poses})

        map_info['outgoing_lanes'] = outgoing_lane_data
        
        return map_info


    def get_patch_coord(self,patch_box: Tuple[float, float, float, float],
                        rotate_center: Tuple[float, float],
                        patch_angle: float = 0.0):
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(rotate_center[0], rotate_center[1]), use_radians=False)

        return patch

    def in_my_patch(self, pos, my_patch):
        if pos[0] > my_patch[0] and pos[1] > my_patch[1] and pos[0] < my_patch[2] and pos[1] < my_patch[3]:
            return True
        else:
            return False

    def in_shapely_polygon(self, pos, polygon: Polygon):
        p = Point(pos[0], pos[1])

        return p.within(polygon)

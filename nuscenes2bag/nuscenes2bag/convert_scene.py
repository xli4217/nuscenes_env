from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped
from matplotlib import pyplot as plt
from nav_msgs.msg import OccupancyGrid, Odometry
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pprint import pprint
from pypcd import numpy_pc2, pypcd
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, NavSatFix, PointCloud2, PointField
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage
from typing import List, Tuple, Dict
from visualization_msgs.msg import ImageMarker, Marker, MarkerArray
from PIL import Image
import cloudpickle
import torch

import math
import numpy as np
import os
import random
import rosbag
import rospy

from nuscenes2bag.utils import Utils
from nuscenes2bag.bitmap import BitMap

from pathlib import Path

def load_experiment_rollout_data(experiment_path: str):
    experiment_result_dict = {
        'figures':{
            'bev/bev': {},
            'ctrl/ctrl': {},
            'input_images/past_raster':{},
            'q_transitions/q_transitions':{}
        },
        'sim_ego_pos': {},
        'sim_ego_quat': {}
    }
    
    #### figures ####
    for img_folder_name in experiment_result_dict['figures'].keys():
        p_list = np.array([str(p) for p in Path(experiment_path+'/'+img_folder_name).glob('*.png')])
        idx = np.argsort([int(p.split('/')[-1][:2]) for p in p_list])
        p_list = p_list[idx].tolist()
        
        for p in p_list:
            with open(p, 'rb') as png_file:
                png = png_file.read()
                idx = int(p.split('/')[-1][:2])
                experiment_result_dict['figures'][img_folder_name][idx] = png
    
    #### ego ####
    d = cloudpickle.load(open(os.path.join(experiment_path, 'scene-0061.pkl'), 'rb'))
    sim_ego_pos = {}
    sim_ego_quat = {}
    for obs in d['obs']:
        sim_ego_pos[obs['sample_idx']] = obs['sim_ego_pos_gb'].tolist() + [0.01] 
        sim_ego_quat[obs['sample_idx']] = obs['sim_ego_quat_gb'].tolist()
    
    experiment_result_dict['sim_ego_pos'] = sim_ego_pos
    experiment_result_dict['sim_ego_quat'] = sim_ego_quat
    
    return experiment_result_dict
            
def convert_scene(scene, utils, dataroot,  *args, **kwargs):
    experiment_result_dict = None
    if 'experiment_result_dict' in kwargs.keys():
        experiment_result_dict = kwargs['experiment_result_dict']
    
    scene_name = scene['name']
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    print(f'Loading map "{location}"')
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)
    print(f'Loading bitmap "{nusc_map.map_name}"')
    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    print(f'Loaded {bitmap.image.shape} bitmap')

    cur_sample = nusc.get('sample', scene['first_sample_token'])

    can_parsers = [
        [nusc_can.get_messages(scene_name, 'ms_imu'), 0, utils.get_imu_msg],
        [nusc_can.get_messages(scene_name, 'pose'), 0, utils.get_odom_msg],
        [nusc_can.get_messages(scene_name, 'steeranglefeedback'), 0, lambda x: utils.get_basic_can_msg('Steering Angle', x)],
        [nusc_can.get_messages(scene_name, 'vehicle_monitor'), 0, lambda x: utils.get_basic_can_msg('Vehicle Monitor', x)],
        [nusc_can.get_messages(scene_name, 'zoesensors'), 0, lambda x: utils.get_basic_can_msg('Zoe Sensors', x)],
        [nusc_can.get_messages(scene_name, 'zoe_veh_info'), 0, lambda x: utils.get_basic_can_msg('Zoe Vehicle Info', x)],
    ]

    bag_name = f'NuScenes-{NUSCENES_VERSION}-{scene_name}.bag'
    bag_path = os.path.join(os.path.abspath(os.curdir), bag_name)
    print(f'Writing to {bag_path}')
    bag = rosbag.Bag(bag_path, 'w', compression='lz4')

    stamp = utils.get_time(nusc.get('ego_pose', nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])['ego_pose_token']))
    map_msg = utils.get_scene_map(scene, nusc_map, bitmap, stamp)
    centerlines_msg = utils.get_centerline_markers(scene, nusc_map, stamp)
    bag.write('/map', map_msg, stamp)
    bag.write('/semantic_map', centerlines_msg, stamp)
    last_map_stamp = stamp

    idx = 0
    while cur_sample is not None:
        if idx > 30:
            break
        sample_lidar = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_lidar['ego_pose_token'])
        stamp = utils.get_time(ego_pose)
        
        #### write experiment_result to bag ####
        if experiment_result_dict is not None:
            # figures #
            for k,v in experiment_result_dict['figures'].items():
                if idx in v.keys():
                    png = v[idx]
                    msg = CompressedImage()
                    msg.header.frame_id = k
                    msg.header.stamp = stamp
                    msg.format = 'png'
                    msg.data = png
                    bag.write("/"+k+'/image_rect_compressed', msg, stamp)
                
            if idx in experiment_result_dict['sim_ego_pos'].keys():
                ego_pose = {
                    'translation': experiment_result_dict['sim_ego_pos'][idx],
                    'rotation': experiment_result_dict['sim_ego_quat'][idx],
                    'token': ego_pose['token'],
                    'timestamp': ego_pose['timestamp']
                }
                
                if idx == list(experiment_result_dict['sim_ego_pos'].keys())[-1]:
                    next_idx = idx
                    break
                else:
                    next_idx = idx+1
                next_sample = utils.nusc.get('sample',cur_sample['next'])
                sample_lidar = utils.nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
                real_next_ego_pose = utils.nusc.get('ego_pose', sample_lidar['ego_pose_token'])

                next_ego_pose = {
                    'translation': experiment_result_dict['sim_ego_pos'][next_idx],
                    'rotation': experiment_result_dict['sim_ego_quat'][next_idx],
                    'token': real_next_ego_pose['token'],
                    'timestamp': real_next_ego_pose['timestamp']
                }
                # publish /tf
                tf_array = utils.get_tfmessage(cur_sample, current_pose=ego_pose, next_pose=next_ego_pose)
                bag.write('/tf', tf_array, stamp)
                
            else:
                # publish /tf
                tf_array = utils.get_tfmessage(cur_sample)
                bag.write('/tf', tf_array, stamp)
        else:
            # publish /tf
            tf_array = utils.get_tfmessage(cur_sample)
            bag.write('/tf', tf_array, stamp)
            
        
        # write map topics every two seconds
        if stamp - rospy.Duration(2.0) >= last_map_stamp:
            map_msg.header.stamp = stamp
            for marker in centerlines_msg.markers:
                marker.header.stamp = stamp
            bag.write('/map', map_msg, stamp)
            bag.write('/semantic_map', centerlines_msg, stamp)
            last_map_stamp = stamp

        # write CAN messages to /pose, /odom, and /diagnostics
        can_msg_events = []
        for i in range(len(can_parsers)):
            (can_msgs, index, msg_func) = can_parsers[i]
            while index < len(can_msgs) and utils.get_utime(can_msgs[index]) < stamp:
                can_msg_events.append(msg_func(can_msgs[index]))
                index += 1
                can_parsers[i][1] = index
        can_msg_events.sort(key = lambda x: x[0])
        for (msg_stamp, topic, msg) in can_msg_events:
            bag.write(topic, msg, stamp)

        
        # /driveable_area occupancy grid
        utils.write_occupancy_grid(bag, nusc_map, ego_pose, stamp)
                
        # iterate sensors
        for (sensor_id, sample_token) in cur_sample['data'].items():
            sample_data = nusc.get('sample_data', sample_token)
            topic = '/' + sensor_id

            # write the sensor data
            # if sample_data['sensor_modality'] == 'radar':
            #     msg = utils.get_radar(sample_data, sensor_id)
            #     bag.write(topic, msg, stamp)
            # elif sample_data['sensor_modality'] == 'lidar':
            #     msg = utils.get_lidar(sample_data, sensor_id)
            #     bag.write(topic, msg, stamp)
            if sample_data['sensor_modality'] == 'camera':
                msg = utils.get_camera(sample_data, sensor_id, dataroot)
                bag.write(topic + '/image_rect_compressed', msg, stamp)
                msg = utils.get_camera_info(sample_data, sensor_id)
                bag.write(topic + '/camera_info', msg, stamp)

            # if sample_data['sensor_modality'] == 'camera':
            #     msg = utils.get_lidar_imagemarkers(sample_lidar, sample_data, sensor_id)
            #     bag.write(topic + '/image_markers_lidar', msg, stamp)
            #     utils.write_boxes_imagemarkers(bag, cur_sample['anns'], sample_data, sensor_id, topic, stamp)

        # publish /pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'base_link'
        pose_stamped.header.stamp = stamp
        pose_stamped.pose.orientation.w = 1
        bag.write('/pose', pose_stamped, stamp)

        # publish /gps
        coordinates = utils.derive_latlon(location, ego_pose)
        gps = NavSatFix()
        gps.header.frame_id = 'base_link'
        gps.header.stamp = stamp
        gps.status.status = 1
        gps.status.service = 1
        gps.latitude = coordinates['latitude']
        gps.longitude = coordinates['longitude']
        gps.altitude = utils.get_transform(ego_pose).translation.z
        bag.write('/gps', gps, stamp)

        # publish /markers/annotations
        marker_array = MarkerArray()
        for annotation_id in cur_sample['anns']:
            ann = nusc.get('sample_annotation', annotation_id)
            marker_id = int(ann['instance_token'][:4], 16)
            c = np.array(nusc.explorer.get_color(ann['category_name'])) / 255.0

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.id = marker_id
            marker.text = ann['instance_token'][:4]
            marker.type = Marker.CUBE
            marker.pose = utils.get_pose(ann)
            marker.frame_locked = True
            marker.scale.x = ann['size'][1]
            marker.scale.y = ann['size'][0]
            marker.scale.z = ann['size'][2]
            marker.color = utils.make_color(c, 0.5)
            marker_array.markers.append(marker)
        
        # ego marker #
        # marker = Marker()
        # marker.header.frame_id = 'map'
        # marker.header.stamp = stamp
        # marker.id = marker_id
        # marker.text = 'ego'
        # marker.type = Marker.CUBE
        # marker.pose = utils.get_pose(ego_pose)
        # marker.frame_locked = True
        # marker.scale.x = 4.3
        # marker.scale.y = 1.8
        # marker.scale.z = 1.6
        # c = [0.1,0.1,0.6]
        # marker.color = utils.make_color(c, 0.9)
        # marker_array.markers.append(marker)
                
        bag.write('/markers/annotations', marker_array, stamp)

        # collect all sensor frames after this sample but before the next sample
        non_keyframe_sensor_msgs = []
        for (sensor_id, sample_token) in cur_sample['data'].items():
            topic = '/' + sensor_id

            next_sample_token = nusc.get('sample_data', sample_token)['next']
            while next_sample_token != '':
                next_sample_data = nusc.get('sample_data', next_sample_token)
                # if next_sample_data['is_key_frame'] or get_time(next_sample_data).to_nsec() > next_stamp.to_nsec():
                #     break
                if next_sample_data['is_key_frame']:
                    break

                # if next_sample_data['sensor_modality'] == 'radar':
                #     msg = utils.get_radar(next_sample_data, sensor_id)
                #     non_keyframe_sensor_msgs.append((msg.header.stamp.to_nsec(), topic, msg))
                # elif next_sample_data['sensor_modality'] == 'lidar':
                #     msg = utils.get_lidar(next_sample_data, sensor_id)
                #     non_keyframe_sensor_msgs.append((msg.header.stamp.to_nsec(), topic, msg))
                if next_sample_data['sensor_modality'] == 'camera':
                    msg = utils.get_camera(next_sample_data, sensor_id, dataroot)
                    camera_stamp_nsec = msg.header.stamp.to_nsec()
                    non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_rect_compressed', msg))

                    msg = utils.get_camera_info(next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/camera_info', msg))

                    # closest_lidar = utils.find_closest_lidar(cur_sample['data']['LIDAR_TOP'], camera_stamp_nsec)
                    # if closest_lidar is not None:
                    #     msg = utils.get_lidar_imagemarkers(closest_lidar, next_sample_data, sensor_id)
                    #     non_keyframe_sensor_msgs.append((msg.header.stamp.to_nsec(), topic + '/image_markers_lidar', msg))
                    # else:
                    #     msg = utils.get_remove_imagemarkers(sensor_id, 'LIDAR_TOP', msg.header.stamp)
                    #     non_keyframe_sensor_msgs.append((msg.header.stamp.to_nsec(), topic + '/image_markers_lidar', msg))

                    # Delete all image markers on non-keyframe camera images
                    # msg = get_remove_imagemarkers(sensor_id, 'LIDAR_TOP', msg.header.stamp)
                    # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_lidar', msg))
                    # msg = get_remove_imagemarkers(sensor_id, 'annotations', msg.header.stamp)
                    # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_annotations', msg))

                next_sample_token = next_sample_data['next']
        
                
        # sort and publish the non-keyframe sensor msgs
        non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
        for (_, topic, msg) in non_keyframe_sensor_msgs:
            bag.write(topic, msg, msg.header.stamp)

        # move to the next sample
        cur_sample = nusc.get('sample', cur_sample['next']) if cur_sample.get('next') != '' else None
        idx += 1
    bag.close()
    print(f'Finished writing {bag_name}')
    
    
if __name__ == "__main__":
    NUSCENES_VERSION = 'v1.0-mini'

    dataroot = os.path.join(os.environ['PKG_PATH'],'data')
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=dataroot, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=dataroot)
    
    utils = Utils(nusc, nusc_can)
    
    scene_data_p = os.path.join(os.environ['PKG_PATH'], 'data', 'supercloud_data', 'CnnLstmAgn_dmp_trainable', 'scene-0061')
    experiment_result_dict = load_experiment_rollout_data(scene_data_p)
    
    convert_scene(nusc.scene[0], utils, dataroot, experiment_result_dict=experiment_result_dict)

    # for scene in nusc.scene:
    #     convert_scene(scene)
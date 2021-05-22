import json
from future.utils import viewitems
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import cv2
import io
import seaborn as sns
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
import time
from pathlib import Path

from .nuscenes_agent import NuScenesAgent

from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
import descartes
import cloudpickle

from utils.configuration import Configuration
from celluloid import Camera
from paths import scene_img_dir 

viz_dir = os.path.dirname(os.path.realpath(__file__))

cars = [plt.imread(viz_dir + '/icons/other_cars.png'),
        plt.imread(viz_dir + '/icons/Car TOP_VIEW 375397.png'),
        plt.imread(viz_dir +'/icons/Car TOP_VIEW F05F78.png'),
        plt.imread(viz_dir +'/icons/Car TOP_VIEW 80CBE5.png'),
        plt.imread(viz_dir +'/icons/Car TOP_VIEW ABCB51.png'),
        plt.imread(viz_dir +'/icons/Car TOP_VIEW C8B0B0.png')]

robot = plt.imread(viz_dir +'/icons/current_car.png')

ped = plt.imread(viz_dir + '/icons/ped.png')


class SceneGraphics(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):
        #### common setup ####
        self.config = Configuration(
            {
                'NuScenesAgent_config': {}
            }
        )
        self.config.update(config)

        super().__init__(config=self.config['NuScenesAgent_config'],
                                            helper=helper,
                                            py_logger=py_logger,
                                            tb_logger=tb_logger)
        self.name = 'SceneGraphics'
        #######
        self.map_layers = [
            'road_divider',
            'lane_divider',
            'drivable_area',
            #'road_segment',
            #'road_block',
            'lane',
            #'ped_crossing',
            'walkway',
            #'stop_line',
            #'carpark_area',
            #'traffic_light'
        ]

        self.plot_list = ['ego', 'other_cars', 'pedestrian', 'cam',
                          'labeled_map', 'sensing_patch', 'sensor_info']
    
    def update_all_info(self):
        pass


    def make_video_from_images(self, image_dir:str=None, video_save_dir:str=None, video_layout=None):
        if video_layout is None:
            video_layout = {
                'figsize': (15,15),
                'nb_rows': 6,
                'nb_cols': 6,
                'components': {
                    'birdseye': [[0,4], [0,6]],
                    'camera': [[4,6], [0,6]]
                }
            }
        
        img_fn_list = [str(p).split('/')[-1] for p in Path(image_dir).rglob('*.png')]

        component_img_list = {}
        for k, v in video_layout['components'].items():
            img_list = [p for p in img_fn_list if k in p and 'checkpoint' not in p]
            idx = np.argsort(np.array([int(p[:2]) for p in img_list]))
            img_list = np.array(img_list)[idx]
            nb_images = len(img_list)
            component_img_list[k] = img_list
             
 
        fig = plt.figure(figsize=video_layout['figsize'], constrained_layout=False)
        gs = fig.add_gridspec(nrows=video_layout['nb_rows'], ncols=video_layout['nb_cols'], wspace=0.01)
        axes = {}
        for k, v in video_layout['components'].items():
            ax = fig.add_subplot(gs[v[0][0]:v[0][1], v[1][0]:v[1][1]])
            ax.axis('off')
            axes[k] = ax
            
        camera = Camera(fig)
            
        for i in tqdm.tqdm(range(nb_images)):
            for k, v in component_img_list.items():
                axes[k].imshow(plt.imread(os.path.join(image_dir, v[i])))
            camera.snap()

        animation = camera.animate()

        if video_save_dir is not None:
            animation.save(video_save_dir+'/video.mp4')
        return animation

        
    def plot_ego_scene(self,
                       ego_centric=True,
                       sample_token:str=None,
                       instance_token:str=None,
                       scene_token:str=None,
                       idx:str="",
                       save_pkl_dir:str=None,
                       save_img_dir:str=None,
                       sensor_info=None,
                       text_box=False,
                       plot_list=None,
                       ego_traj=None,
                       ado_traj=None,
                       contour=None,
                       read_from_cached=False,
                       paper_ready=False,
                       other_images_to_be_saved=None,
                       render_additional=None
    ):

        '''
        ego_traj = {
           <name>: {
            'traj': <traj>,
            'color': <color>
          }
        }

        ado_traj = {
          <instance_token>: {
             "traj_dist": [[mean, cov], [], ...],
             "frame": <"local" or "global">,
             "pos": <np.ndarray or list>, # global coordinate of origin
             "quat": <np.ndarray or list> # global rotation of origin
          }
        }
        '''

        if sample_token is not None and scene_token is not None:
            raise ValueError("only one of sample_token or scene_token should be provided")

        if scene_token is not None:
            sample_token = self.nusc.get('scene', scene_token)['first_sample_token']
            
        sample = self.nusc.get('sample', sample_token)
        if plot_list is None:
            plot_list = self.plot_list
        if sensor_info is not None:
            plot_list += ['sensing_patch']
            
        fig, ax, other = self.plot_agent_scene(ego_centric=ego_centric,
                                               sample_token=sample_token,
                                               instance_token=instance_token,
                                               sensor_info=sensor_info,
                                               text_box=text_box,
                                               plot_list=plot_list,
                                               ego_traj=ego_traj,
                                               read_from_cached=read_from_cached,
                                               paper_ready=paper_ready,
                                               render_additional=render_additional
        )


        #### plot sim ego ####
        if ego_traj is not None:
            if 'sim_ego' in ego_traj.keys():
                sim_ego = ego_traj['sim_ego']
                if sim_ego is not None:
                    self.plot_elements(sim_ego['pos'], sim_ego['yaw'], 'sim_ego', ax, animated_agent=paper_ready)

        #### plot ado traj ####
        if ado_traj is not None:
            self.plot_trajectory_distributions(ax, ado_traj)

        #### plot contour ####
        if contour is not None:
            self.plot_contour(ax, contour)

        #### save stuff ####
        if save_pkl_dir is not None:
            with open(p, 'wb') as f:
                p = os.path.join(save_pkl_dir, idx+"_"+sample_token+".pkl")
                cloudpickle.dump(ax, p)

        if save_img_dir is not None:
            p = os.path.join(save_img_dir, idx+"_"+sample_token+"_birdseye.png")
            #fig.savefig(p, dpi=300, quality=95)
            fig.savefig(p)
            if 'cam' in plot_list:
                p = os.path.join(save_img_dir, idx+"_"+sample_token+"_camera.png")
                #other['sfig'].savefig(p, dpi=300, quality=95)
                other['sfig'].savefig(p)

            if other_images_to_be_saved is not None:
                for k, v in other_images_to_be_saved.items():
                    p = os.path.join(save_img_dir, idx+"_"+sample_token+"_"+k+".png")
                    if isinstance(v, np.ndarray):
                        plt.imsave(p, v)
                    elif isinstance(v, matplotlib.figure.Figure):
                        plt.savefig()

        return fig, ax

    def plot_agent_scene(self,
                         ego_centric:bool=False,
                         ego_traj=None,
                         instance_token:str=None,
                         sample_token:str=None,
                         map_layers=None,
                         sensor_info=None,
                         sensing_patch=None,
                         predictions=None,
                         agent_state_dict=None,
                         plot_list=None,
                         legend=False,
                         text_box=False,
                         show_axis=True,
                         render_ego_pose_range=False,
                         paper_ready=False,
                         read_from_cached=False,
                         plot_agent_trajs=True,
                         animated_agent=False,
                         bfig=None, # for birdseye image
                         bax=None,
                         sfig=None, # for camera
                         sax=None,
                         render_additional=None
    ):

        if paper_ready:
            legend = False
            text_box = False
            show_axis = False
            render_ego_pose_range = False
            plot_agent_trajs=False
            animated_agent=True

        if map_layers is None:
            map_layers = self.map_layers

        if plot_list is None:
            plot_list = self.plot_list

        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        scene_log = self.nusc.get('log', scene['log_token'])
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=scene_log['location'])

        patch_margin = 30
        min_diff_patch = 30

        if not ego_centric:
            # if sensor_info is not None:
            #     agent_future = sensor_info['agent_info']['current_agent']['future']
            #     agent_past = sensor_info['agent_info']['current_agent']['past']
            # else:
            agent_future = self.helper.get_future_for_agent(instance_token,
                                                            sample_token,
                                                            self.na_config['pred_horizon'],
                                                            in_agent_frame=False,
                                                            just_xy=True)

            agent_past = self.helper.get_past_for_agent(instance_token,
                                                        sample_token,
                                                        self.na_config['obs_horizon'],
                                                        in_agent_frame=False,
                                                        just_xy=True)

            #### set plot patch ####

            min_patch = np.floor(agent_future.min(axis=0) - patch_margin)
            max_patch = np.ceil(agent_future.max(axis=0) + patch_margin)
            diff_patch = max_patch - min_patch

            if any(diff_patch < min_diff_patch):
                center_patch = (min_patch + max_patch) / 2
                diff_patch = np.maximum(diff_patch, min_diff_patch)
                min_patch = center_patch - diff_patch / 2
                max_patch = center_patch + diff_patch / 2
            my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

        else:
            sample_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            my_patch = (
                ego_pose['translation'][0]-patch_margin,
                ego_pose['translation'][1]-patch_margin,
                ego_pose['translation'][0]+patch_margin,
                ego_pose['translation'][1]+patch_margin,
            )


        #### read from saved path if present ####
        read_img = False
        if read_from_cached:
            scene_path = os.path.join(scene_img_dir, scene['name']+"-token-"+sample['scene_token'])
            p_scene = Path(scene_img_dir)
            saved_scene_list = [str(f) for f in p_scene.iterdir() if f.is_dir()]

            if scene_path in saved_scene_list:
                p_sample = Path(scene_path)
                for f in p_sample.iterdir():
                    if sample_token in str(f):
                        ax = cloudpickle.load(open(os.path.join(scene_path, str(f)), 'rb'))
                        fig = plt.figure(figsize=(10,10))
                        fig._axstack.add('ax', ax)
                        read_img = True
        if not read_img:
            fig, ax = nusc_map.render_map_patch(my_patch,
                                                map_layers,
                                                figsize=(10, 10),
                                                render_egoposes_range=render_ego_pose_range,
                                                render_legend=legend,
                                                fig=bfig,
                                                axes=bax)

      
            if not ego_centric:
                ax.set_title(scene['name']+" instance_token: " + instance_token + ", sample_token: " + sample_token + "\n" + ", decription " + scene['description'])
            else:
                ax.set_title(scene['name']+", sample_token: " + sample_token +"\n"+ ", decription " + scene['description'])

        
        #### label map ####
        if 'labeled_map' in plot_list:
            records_within_patch = nusc_map.get_records_in_patch(my_patch, nusc_map.non_geometric_layers, mode='within')
            self.label_map(ax, nusc_map, records_within_patch['stop_line'], text_box=text_box)

        
            #### Plot ego ####
            if 'ego' in plot_list:
                ego_pos, ego_quat = self.plot_ego(ax, sample, ego_traj=ego_traj, animated_agent=animated_agent)

            
            #### Plot other agents ####
            if 'pedestrian' in plot_list or 'other_cars' in plot_list:
                road_agents_in_patch = self.plot_road_agents(ax,
                                                             instance_token,
                                                             sample,
                                                             plot_list,
                                                             text_box,
                                                             sensor_info,
                                                             my_patch,
                                                             plot_traj=plot_agent_trajs,
                                                             animated_agent=animated_agent
                )

        ##################
        # Car to predict #
        ##################
        if not ego_centric:
            agent_pos, agent_quat = self.plot_car_to_predict(ax,
                                                             agent_future,
                                                             agent_past,
                                                             instance_token,
                                                             sample_token,
                                                             text_box,
                                                             predictions,
                                                             agent_state_dict)

            
        #### plot all_info ####
        if not ego_centric:
            self.plot_map_info(ax, agent_pos, nusc_map, text_box=text_box)
        else:
            self.plot_map_info(ax, ego_pos, nusc_map, text_box=text_box)

        #### plot sensor info ###
        if sensor_info is not None and 'sensor_info' in plot_list:
            self.plot_sensor_info(ax, sensor_info=sensor_info, text_box=text_box)

        #### render map layers on camera images ####
        if 'cam' in plot_list:
            #sfig, sax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9,16))        
                
            layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
            layer_names = []
            #cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            #             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
            cam_names = ['CAM_FRONT']
            k = 0
            if sfig is None:
                if len(cam_names) == 6:
                    sfig, sax = plt.subplots(nrows=2, ncols=3)
                    for i in range(2):
                        for j in range(3):
                            sax[i,j].xaxis.set_visible(False)
                            sax[i,j].yaxis.set_visible(False)
                            cam_fig, cam_ax = nusc_map.render_map_in_image(self.nusc, sample_token, layer_names=layer_names, camera_channel=cam_names[k], ax=sax[i,j])
                            k += 1
                elif len(cam_names) == 1:
                    sfig, sax = plt.subplots()
                    sax.xaxis.set_visible(False)
                    sax.yaxis.set_visible(False)
                    cam_fig, cam_ax = nusc_map.render_map_in_image(self.nusc, sample_token, layer_names=layer_names, camera_channel=cam_names[k], ax=sax)
                else:
                    raise ValueError('')
                    
            sfig.tight_layout(pad=0)
            sfig.set_figheight(7)
            sfig.set_figwidth(15)
            
            # for car_info in road_agents_in_patch['vehicles']:
            #     instance_token = car_info['instance_token']
            #     # render annotations inside patch
            #     ann = self.helper.get_sample_annotation(instance_token, sample_token)
            #     ann_fig, ann_ax = self.nusc.render_annotation(ann['token'])
            #     if sensing_patch is not None and self.in_shapely_polygon(car_info['translation'], sensing_patch):
            #         ann_ax.set_title("Sensed")
        else:
            sfig, sax = None, None
            cam_fig, cam_ax = None, None

        #### render additional outside information ####
        if render_additional is not None:
            self.render_additional(ax, render_additional)
            
        if not show_axis:
            plt.axis('off')
            plt.grid('off')
            ax.grid(False)
        ax.set_aspect('equal')
      

        other = {'cam_fig': cam_fig, 'cam_ax': cam_ax, 'sfig': sfig, 'sax': sax}
        return fig, ax, other

    def render_additional(self, ax, render_dict:dict=None):
        if 'lines' in render_dict.keys():
            # lines = [
            #     {
            #         'start': <2x1 vector>,
            #         'end': <2x1 vector>
            #         'color': <color>
            #     },
            #     {
            #          'traj': <2xn vector>,
            #          'color': <color>,
            #          'marker': <marker>
            #     }
            # ]
            for l in render_dict['lines']:
                if 'start' in l.keys():
                    ax.plot([l['start'][0], l['end'][0]], [l['start'][1], l['end'][1]], c=l['color'])
                elif 'traj' in l.keys():
                    ax.plot(l['traj'][:,0], l['traj'][:,1], c=l['color'], linestyle=l['marker'])

        if 'scatters' in render_dict.keys():
            # scatters = [
            #     {
            #         'traj': <nx2 matrix>,
            #         'color': <color>
            #     }
            # ]
            for s in render_dict['scatters']:
                ax.scatter(s['traj'][:,0], s['traj'][:,1], color=s['color'], s=30, zorder=700)

    def in_my_patch(self, pos, my_patch):
        if pos[0] > my_patch[0] and pos[1] > my_patch[1] and pos[0] < my_patch[2] and pos[1] < my_patch[3]:
            return True
        else:
            return False


    def plot_center_lanes(self):
        pass

    def plot_elements(self,
                      pos: np.ndarray,
                      heading: float,
                      object_type="current_car",
                      ax=None,
                      label: str="",
                      attribute: str = "",
                      animated_agent=False
    ):

        '''pos is the global coordinate of the object
           heading is in degrees
           object_type can be 'current_car' 'other_car', 'pedestrian'
        '''
        if object_type == 'current_car':
            obj = robot
        elif object_type == 'other_cars':
            obj = cars[0]
        elif object_type == 'ego':
            obj = cars[4]
        elif object_type == 'sim_ego':
            obj = cars[2]
        elif object_type == 'pedestrian':
            obj = ped
        else:
            raise ValueError('object type not supported')

        if object_type != 'pedestrian':
            r_img = rotate(obj, angle=90, axes=(1,0))
            r_img = rotate(r_img, angle=-heading, axes=(1,0))
            if object_type == 'current_car':
                oi = OffsetImage(r_img, zoom=0.02, zorder=700)
                color='green'
            elif object_type == 'other_cars' :
                oi = OffsetImage(r_img, zoom=0.035, zorder=700)
                color = 'blue'
            elif object_type == 'ego':
                oi = OffsetImage(r_img, zoom=0.015, zorder=700)
                color = 'red'
            elif object_type == 'sim_ego':
                oi = OffsetImage(r_img, zoom=0.015, zorder=700)
                color = 'yellow'            
                
            veh_box = AnnotationBbox(oi, (pos[0], pos[1]), frameon=False)
            veh_box.zorder = 700
            if animated_agent:
                ax.add_artist(veh_box)
            else:
                ax.scatter(pos[0], pos[1], marker='H', color=color, s=100, zorder=700)
        else:
            ax.scatter(pos[0], pos[1], marker='*', color='green', s=100)

    def plot_text_box(self, ax, text_string:str, pos: np.ndarray, facecolor: str='wheat'):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(pos[0], pos[1], text_string, fontsize=10, bbox=props)


    def plot_contour(self, ax, contour):
        X = contour['X']
        Y = contour['Y']
        Z = contour['Z']
        levels = contour['levels']
        transform = contour['transform']

        X_global = X
        Y_global = Y
        # if transform is not None:
        #     Coord_local = np.concatenate([np.expand_dims(X, axis=0),
        #                                   np.expand_dims(Y, axis=0)], axis=0)
        #     coord_global = convert_local_coords_to_global(Coord_local.reshape(2, -1).T, transform['translation'], transform['rotation'])
            
        #     X_global = []
        #     Y_global = []
        #     for i in range(0, coord_global.shape[0], X.shape[1]):
        #         X_global.append(coord_global[i:i+X.shape[1], 0].tolist())
        #         Y_global.append(coord_global[i:i+X.shape[1], 1].tolist())
        #     X_global = np.array(X_global)
        #     Y_global = np.array(Y_global)    

        cp = ax.contourf(X_global, Y_global, Z, levels, zorder=100, alpha=0.5, cmap='Reds',linewidths=3)

    def plot_sensor_info(self, ax, sensor_info,text_box=True, plot_ado_connection_lines=False):
        #### plot sensing patch ####
        sensing_patch = sensor_info['sensing_patch']['polygon']
        polygon  = matplotlib.patches.Polygon(np.array(list(sensing_patch.exterior.coords)),
                                              fill=True,
                                              fc='green',
                                              alpha=0.3,
                                              #edgecolor='green',
                                              #linestyle='--',
                                              linewidth=2)

        ax.add_patch(polygon)

        
        #### plot ego ####
        ego_info = sensor_info['ego_info']
        ego_pos = ego_info['translation'][:2]
        ego_quat = ego_info['rotation_quat']
        
        #### plot agents ####
        agent_info = sensor_info['agent_info']
        if plot_ado_connection_lines:
            for agent in agent_info:
                agent_pos = agent['translation'][:2]
                ax.plot([ego_pos[0], agent_pos[0]], [ego_pos[1], agent_pos[1]], c='black')
        
        #### plot map info ####
        
    def plot_map_info(self, ax, agent_pos, nusc_map, text_box=True):
        closest_lane_id = nusc_map.get_closest_lane(agent_pos[0], agent_pos[1], radius=2)
        closest_lane_record = nusc_map.get_lane(closest_lane_id)

        closest_lane_poses = np.array(arcline_path_utils.discretize_lane(closest_lane_record, resolution_meters=1))

        
        incoming_lane_ids = nusc_map.get_incoming_lane_ids(closest_lane_id)
        incoming_lane_data = []
        for incoming_lane_id in incoming_lane_ids:
            i_record = nusc_map.get_lane(incoming_lane_id)
            i_poses = np.array(arcline_path_utils.discretize_lane(i_record, resolution_meters=1))
            incoming_lane_data.append({'record': i_record, 'poses': i_poses})

            
        outgoing_lane_ids = nusc_map.get_outgoing_lane_ids(closest_lane_id)
        outgoing_lane_data = []
        for outgoing_lane_id in outgoing_lane_ids:
            o_record = nusc_map.get_lane(outgoing_lane_id)
            o_poses = np.array(arcline_path_utils.discretize_lane(o_record, resolution_meters=1))
            outgoing_lane_data.append({'record': o_record, 'poses': o_poses})


        map_info = {
                'closest_lane': {'record': closest_lane_record, 'poses': closest_lane_poses},
                'incoming_lanes': incoming_lane_data,
                'outgoing_lanes': outgoing_lane_data
        }
       

        for k, v in viewitems(map_info):
            if k == 'stop_line':
                for d in v:
                    bd = d['bounding_box']
                    center = [(bd[0]+bd[2])/2, (bd[1]+bd[3])/2]
                    if text_box:
                        self.plot_text_box(ax, 'detected_'+d['record']['stop_line_type'], center, 'blue')
                    
            elif k == 'closest_lane':
                p = np.array(v['poses'])
                ax.plot(p[:, 0], p[:, 1], linestyle="-.", linewidth=2, color='yellow')

            elif k == 'incoming_lanes':
                for d in v:
                    p = np.array(d['poses'])
                    ax.plot(p[:, 0], p[:, 1], linestyle="-.", linewidth=2, color='brown')
            elif k == 'outgoing_lanes':
                for d in v:
                    p = np.array(d['poses'])
                    ax.plot(p[:, 0], p[:, 1], linestyle="-.", linewidth=2, color='white')
            else:
                raise ValueError(f'info type {k} not supported')

    
    def plot_car_to_predict(self,
                            ax,
                            agent_future: np.ndarray,
                            agent_past: np.ndarray,
                            instance_token: str,
                            sample_token: str,
                            text_box: bool=True,
                            predictions: list=None,
                            agent_state_dict:dict=None
    ):

        '''
        predictions = {
            'name': {'data': <data>, 'color': <coloar>, 'frame': <frame>, 'style':'.'}
        }
        
        '''
        
        ## plot car ####
        ann = self.helper.get_sample_annotation(instance_token, sample_token)
        
        category = ann['category_name']
        if len(ann['attribute_tokens']) != 0:
            attribute = self.nusc.get('attribute', ann['attribute_tokens'][0])['name']
        else:
            attribute = ""
            
        agent_yaw = Quaternion(ann['rotation'])
        agent_yaw = quaternion_yaw(agent_yaw)
        agent_yaw = angle_of_rotation(agent_yaw)
        agent_yaw = np.rad2deg(agent_yaw)

        self.plot_elements([ann['translation'][0], ann['translation'][1]], agent_yaw, 'current_car', ax)
        if text_box:
            self.plot_text_box(ax, category, [ann['translation'][0]+1.2, ann['translation'][1]])
            self.plot_text_box(ax, attribute, [ann['translation'][0]+1.2, ann['translation'][1]+1.2])
            if agent_state_dict is not None:
                state_str = ""
                for k, v in agent_state_dict.items():
                    state_str += f"{k[0]}:{v:.2f}, "
                self.plot_text_box(ax, state_str, [ann['translation'][0]+1.2, ann['translation'][1]+3.2])
                                                            
        agent_pos = [ann['translation'][0], ann['translation'][1]]
        agent_yaw_deg = agent_yaw


        # plot ground truth
        if len(agent_future) > 0:
            ax.scatter(agent_future[:, 0], agent_future[:, 1], s=20, c='yellow', alpha=1.0, zorder=200)
        if len(agent_past) > 0:
            ax.scatter(agent_past[:, 0], agent_past[:, 1], s=20, c='k', alpha=0.5, zorder=200)

        # plot predictions
        if predictions is not None:
            for k, v in viewitems(predictions):
                if v['frame'] == 'local':
                    v['data'] = convert_local_coords_to_global(v['data'], np.array(agent_pos), np.array(ann['rotation']))
                if 'style' not in v.keys():
                    v['style'] = '.'
                if v['style'] == '.':
                    ax.scatter(v['data'][:, 0], v['data'][:, 1], s=20, c=v['color'], alpha=1.0, zorder=2)
                elif v['style'] == '-':
                    ax.plot(v['data'][:, 0], v['data'][:, 1], c=v['color'], alpha=1.0, zorder=2)
                else:
                    raise ValueError('style not supported')
    
        
        return agent_pos, ann['rotation']
    
    def label_map(self, ax, nusc_map, map_records, text_box=True):
        #### Label map ####
        for record_token in map_records:
            bd = nusc_map.get_bounds('stop_line', record_token)
            center = [(bd[0]+bd[2])/2, (bd[1]+bd[3])/2]
            record = nusc_map.get('stop_line', record_token)
            stop_line_type = record['stop_line_type']
            if text_box:
                self.plot_text_box(ax, stop_line_type, center, 'white')


    def plot_ego(self, ax, sample, ego_traj=None, animated_agent=False):
        sample_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            
        pos = [ego_pose['translation'][0], ego_pose['translation'][1]]
        ego_yaw = Quaternion(ego_pose['rotation'])
        ego_yaw = quaternion_yaw(ego_yaw)
        ego_yaw = angle_of_rotation(ego_yaw)
        ego_yaw = np.rad2deg(ego_yaw)

        self.plot_elements(pos, ego_yaw, 'ego', ax, animated_agent=animated_agent)

        if ego_traj is not None:
            for name, traj_dict in ego_traj.items():
                ax.scatter(traj_dict['traj'][:,0], traj_dict['traj'][:,1], c=traj_dict['color'], s=60, zorder=80)
                    
        return pos, ego_pose['rotation']

    def plot_trajectory_distributions(self, ax, traj_dist_dict):
        '''
        traj_dist_dict = {
            <instance_sample_token>: {"traj_dist": [[mean, cov], [], ...],
                                      "frame": <"local" or "global">,
                                      "pos": <np.ndarray or list>, # global coordinate of origin
                                      "quat": <np.ndarray or list> # global rotation of origin

                                      }
        }
    
        '''
        for k, v in traj_dist_dict.items():
            traj_dist = v['traj_dist']
            for dist in traj_dist:
                mean, cov = dist
                if v['frame'] == 'local':
                    mean = convert_local_coords_to_global(np.array(mean), v['pos'], v['quat'])
                x, y = np.random.multivariate_normal(mean, cov, size=100).T
                sns.kdeplot(x=x,y=y, cmap='Blues', ax=ax)
        
    def plot_road_agents(self,
                         ax,
                         fig,
                         sample,
                         plot_list,
                         text_box,
                         sensor_info,
                         my_patch,
                         plot_traj=False,
                         contour_func=None,
                         animated_agent=False
    ):

        road_agents_in_patch = {
            'pedestrians': [],
            'vehicles': []
        }
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            if len(ann['attribute_tokens']) != 0:
                attribute = self.nusc.get('attribute', ann['attribute_tokens'][0])['name']
            else:
                attribute = ""

            pos = [ann['translation'][0], ann['translation'][1]]
            instance_token = ann['instance_token']
            sample_token = sample['token']
            #### Plot other agents ####
            valid_agent = False
            if 'other_cars' in plot_list and 'vehicle' in category and 'parked' not in attribute and self.in_my_patch(pos, my_patch):
                valid_agent = True
                agent_yaw = Quaternion(ann['rotation'])
                agent_yaw = quaternion_yaw(agent_yaw)
                agent_yaw = angle_of_rotation(agent_yaw)
                agent_yaw = np.rad2deg(agent_yaw)
                self.plot_elements(pos, agent_yaw, 'other_cars', ax, animated_agent=animated_agent)

                car_info = {
                    'instance_token': ann['instance_token'],
                    'category': category,
                    'attribute': attribute,
                    'translation': pos,
                    'rotation_quat': ann['rotation'],
                    'rotation_deg': agent_yaw
                }
                road_agents_in_patch['vehicles'].append(car_info)

                if text_box:
                    self.plot_text_box(ax, category, [ann['translation'][0]+1.2, ann['translation'][1]])
                    self.plot_text_box(ax, attribute, [ann['translation'][0]+1.2, ann['translation'][1]-1.2])
                    
            #### Plot pedestrians ####
            if 'pedestrian' in plot_list and 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category and self.in_my_patch(pos, my_patch):
                valid_agent = True
                agent_yaw = Quaternion(ann['rotation'])
                agent_yaw = quaternion_yaw(agent_yaw)
                agent_yaw = angle_of_rotation(agent_yaw)
                agent_yaw = np.rad2deg(agent_yaw)
               
                self.plot_elements(pos, agent_yaw, 'pedestrian', ax)
                if text_box:
                    self.plot_text_box(ax, category, [ann['translation'][0]+1.2, ann['translation'][1]])
                    self.plot_text_box(ax, attribute, [ann['translation'][0]+1.2, ann['translation'][1]-1.2])
            

            if valid_agent and plot_traj:
                agent_future = self.helper.get_future_for_agent(instance_token,
                                                                sample_token,
                                                                self.na_config['pred_horizon'],
                                                                in_agent_frame=False,
                                                                just_xy=True)
            
                agent_past = self.helper.get_past_for_agent(instance_token,
                                                            sample_token,
                                                            self.na_config['obs_horizon'],
                                                            in_agent_frame=False,
                                                            just_xy=True)

                if len(agent_future) > 0:
                    ax.scatter(agent_future[:, 0], agent_future[:, 1], s=10, c='y', alpha=1.0, zorder=200)
                if len(agent_past) > 0:
                    ax.scatter(agent_past[:, 0], agent_past[:, 1], s=10, c='k', alpha=0.2, zorder=200)

        return road_agents_in_patch

if __name__ == "__main__":
    import os

    cls = SceneGraphics()
    p = image_path=os.environ['PKG_PATH']+'/tmp'
    cls.generate_video_from_images(image_path=p, video_save_path=os.path.join(p, 'video.mp4'))
    

                    

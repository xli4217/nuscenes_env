import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
import numpy as np
import os
from utils.utils import assert_shape

def render(graphics, render_info, config={}):
    
    if 'image' in config['render_type']:
        sim_ego_yaw = Quaternion(render_info['sim_ego_quat_gb'])
        sim_ego_yaw = quaternion_yaw(sim_ego_yaw)
        sim_ego_yaw = angle_of_rotation(sim_ego_yaw)
        sim_ego_yaw = np.rad2deg(sim_ego_yaw)

        ego_traj = None
        if 'sim_ego' in config['render_elements']:
            ego_traj = {
                # 'lane': {
                #     'traj': self.all_info['future_lanes'][0],
                #     'color': 'green'
                # },
                'sim_ego':{
                    'pos': render_info['sim_ego_pos_gb'],
                    'yaw': sim_ego_yaw,
                    'traj': np.zeros((4,2)),
                    'color': 'yellow'
                }
            }

        #### control plots ####
        '''
        if 'control_plots' in config['render_elements']:
            fig, ax = plt.subplots(2,1)
            ax[0].plot(list(ap_timesteps), list(ap_speed), 'o-')
            ax[0].set_xlabel('timestep', fontsize=20)
            ax[0].set_ylabel('speed', fontsize=20)
            ax[1].plot(list(ap_timesteps), list(ap_steering), 'o-')
            ax[1].set_xlabel('timestep', fontsize=20)
            ax[1].set_ylabel('steering', fontsize=20)
            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            if 'image' not in render_info.keys():
                render_info['image'] = {}
            render_info['image'].update({'cmd': image})
        '''

        save_img_dir = None
        if render_info['save_image_dir'] is not None:
            save_img_dir = os.path.join(render_info['save_image_dir'], str(render_info['scene_name']))
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir, exist_ok=True)
                
        sensor_info = None
        if 'sensor_info' in config['render_elements']:
            sensor_info = render_info['all_info']['sensor_info']

        ado_traj_dict = None
        if 'ado_traj_dict' in render_info.keys():
            ado_traj_dict = render_info['ado_traj_dict']

        costmap_contour = None
        if 'costmap_contour' in render_info.keys():
            costmap_contour = render_info['costmap_contour']

        other_images_to_be_saved = None
        if render_info['all_info']['sim_ego_raster_image'] is not None:
            raster =render_info['all_info']['sim_ego_raster_image']
            if raster.shape == (3,250,250):
                raster = np.transpose(raster, (1,2,0))
            assert_shape(raster,'raster', (250,250,3))
            other_images_to_be_saved = {
                'raster': raster
            }

        if 'image' in render_info.keys():
            if other_images_to_be_saved is None:
                other_images_to_be_saved = render_info['image']
            else:
                other_images_to_be_saved.update(render_info['image'])

        render_additional = {}
        if 'lines' in render_info.keys():
            render_additional['lines'] = render_info['lines']
        if 'scatters' in render_info.keys():
            render_additional['scatters'] = render_info['scatters']
        if 'text_boxes' in render_info.keys():
            render_additional['text_boxes'] = render_info['text_boxes']

        if render_info['instance_token'] is None:
            ego_centric = True
        else:
            ego_centric = False

        plot_human_ego = True
        if 'human_ego' not in config['render_elements']:
            plot_human_ego = False

        ego_centric = False
        if render_info['instance_token'] == 'ego' or render_info['instance_token'] is None:
            ego_centric = True

        fig, ax = graphics.plot_ego_scene( 
            ego_centric=ego_centric,
            sample_token=render_info['sample_token'],
            instance_token=render_info['instance_token'],
            ego_traj=ego_traj,
            ado_traj=ado_traj_dict,
            contour=costmap_contour,
            save_img_dir=save_img_dir,
            idx=str(render_info['sample_idx']).zfill(2),
            sensor_info=sensor_info,
            paper_ready=config['render_paper_ready'],
            other_images_to_be_saved=other_images_to_be_saved,
            render_additional = render_additional,
            plot_human_ego=plot_human_ego,
            patch_margin=config['patch_margin'],
        )

        return fig, ax

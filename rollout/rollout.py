import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import os
import shutil
from utils.utils import convert_local_coords_to_global, convert_global_coords_to_local
from loguru import logger

def rollout(scene_name=None, 
            scene_idx=None, 
            sample_idx=0,
            nb_steps=None,
            env=None, 
            policy=None, 
            plot_elements=[],
            debug=False, 
            logger=None,
            scene_image_dir=None,
            demo_goal_termination=False):
    
    if scene_image_dir is not None:
        save_img_dir = os.path.join(scene_image_dir, scene_name)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir, exist_ok=True)

    #env.py_logger = logger
    if scene_name is not None:
        obs = env.reset(scene_name=scene_name, sample_idx=sample_idx)
    if scene_idx is not None:
        obs = env.reset(scene_idx=scene_idx, sample_idx=sample_idx)
    done = False

    policy.reset(obs)

    if scene_name == 'scene-1100':
        right_turing_lane = np.load(os.environ['PKG_PATH']+'/logic_risk_ioc/dataset/scene-1100_right_turning_lane.npz.npy')

    step = 0
    env_info = []
    policy_info = []

    ego_goal_gb = obs['ego_pos_traj'][-1][:2]
    
    while not done:
        print(f"step: {env.sample_idx}")
        render_info = {}
        if scene_name == 'scene-1100':
            obs['gt_future_lanes'] = [right_turing_lane]
            render_info.update({'lines':[{'traj': right_turing_lane, 'color':'yellow', 'marker':'-.'}]})

        ego_goal = convert_global_coords_to_local(ego_goal_gb, obs['sim_ego_pos_gb'], obs['sim_ego_quat_gb'])

        env_info.append(obs)
        action, render_info_env, other_info = policy.get_action(obs, goal=ego_goal)
        policy_info.append(other_info)

        if render_info_env is not None:
            render_info.update(render_info_env)
        
        #### one step ####
        obs, done, other = env.step(action, render_info, save_img_dir=scene_image_dir)

        
        #############
        # Visualize #
        #############
        if 'raster' in plot_elements:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
            raster = obs['sim_ego_raster_image']
            if raster.shape == (3,250,250):
                ax.imshow(np.transpose(raster, (1,2,0)))
            else:
                ax.imshow(raster)
            plt.show()
            
        step += 1
        if debug:
            break
        if nb_steps is not None and step >= nb_steps:
            return env_info, policy_info
        
    return env_info, policy_info



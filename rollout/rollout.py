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
            sample_idx=0, env=None, 
            policy=None, 
            plot_elements=[],
            debug=False, 
            logger=None,
            scene_image_dir=None,
            demo_goal_termination=True):

    if scene_image_dir is not None:
        save_img_dir = os.path.join(scene_image_dir, scene_name)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir, exist_ok=True)


    env.py_logger = logger
    if scene_name is not None:
        obs = env.reset(scene_name=scene_name, sample_idx=sample_idx)
    if scene_idx is not None:
        obs = env.reset(scene_idx=scene_idx, sample_idx=sample_idx)
    done = False
    policy.reset(obs)

    ego_traj = [obs['ego_pos_gb']]
    sim_ego_traj = [obs['sim_ego_pos_gb']]
    policy_info_traj = []
    
    if scene_name == 'scene-1100':
        right_turing_lane = np.load(os.environ['PKG_PATH']+'/logic_risk_ioc/dataset/scene-1100_right_turning_lane.npz.npy')

    step = 0
    dist_to_ados_scene = []
    scene_info = {}
    goal_pos = obs['ego_pos_traj'][-1][:2]

    while not done:
        print(f"step: {step}")
        render_info = {}
        if scene_name == 'scene-1100':
            obs['gt_future_lanes'] = [right_turing_lane]
            render_info.update({'lines':[{'traj': right_turing_lane, 'color':'yellow', 'marker':'-.'}]})

        ego_goal = convert_global_coords_to_local(np.array([obs['ego_pos_traj'][-1][:2]]), obs['sim_ego_pos_gb'], obs['sim_ego_quat_gb'])
        print(f"goal: {ego_goal}")
        action, render_info_env, other_info = policy.get_action(obs, goal=ego_goal)
        policy_info_traj.append(other_info)
        
        if render_info_env is not None:
            render_info.update(render_info_env)

        #### record scene_info ####
        # scene_info = populate_scene_info(scene_info, obs, policy)
        
        #### one step ####
        obs, done, other = env.step(action, render_info)

        ego_traj.append(obs['ego_pos_gb'])
        sim_ego_traj.append(env.sim_ego_pos_gb)
        dist_to_goal = np.linalg.norm(goal_pos - env.sim_ego_pos_gb)

        if demo_goal_termination:
            if dist_to_goal < 20:
                done = True
                print("reach goal")

        agent_info = obs['sensor_info']['agent_info']
        
        #############
        # Visualize #
        #############
        if 'raster' in plot_elements:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
            ax.imshow(np.transpose(obs['sim_ego_raster_image'], (1,2,0)))
            plt.show()
            
        #####################
        # Calculate metrics #
        #####################
        #### calculate minimum distance to nearby ados ####
        dist_to_ados_sample = [100.]
        for a in agent_info:
            dist_to_ado = np.linalg.norm(np.array(a['translation'][:2])-np.array(obs['sim_ego_pos_gb']))
            dist_to_ados_sample.append(dist_to_ado)

        dist_to_ados_scene.append(min(dist_to_ados_sample))
        step += 1
        if debug:
            break

    return np.array(ego_traj), np.array(sim_ego_traj), min(dist_to_ados_scene), scene_info, policy_info_traj



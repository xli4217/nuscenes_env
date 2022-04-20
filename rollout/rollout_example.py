import time
import cloudpickle
from utils.utils import class_from_path
import numpy as np
import matplotlib.pyplot as plt

def get_env(version='v1.0-mini', env_path=None, save_pkl_path=None, render_bev=True, config={}):
    if env_path is not None:
        t = time.time()
        env = cloudpickle.load(open(env_path, 'rb'))    
        print(f"env load time: {time.time()-t}")
    else:
        env_config = config
        env_config['config']['NuScenesAgent_config']['version'] = version    
        env = class_from_path(env_config['type'])(env_config['config'])
    
    if not render_bev:
        env.config['render_type'] = []
    
    if 'pedestrian' in env.graphics.plot_list :
        env.graphics.plot_list.remove('pedestrian')
    if 'map_info' in env.graphics.plot_list:
        env.graphics.plot_list.remove('map_info')
    if 'cam' in env.graphics.plot_list:
        env.graphics.plot_list.remove('cam')

    if save_pkl_path is not None:
        cloudpickle.dump(env, open(save_pkl_path, 'wb'))
        
    return env


def test_rollout(scene_name='scene-0061', 
                 version='v1.0-mini',
                 model=None,
                 scene_pkl_dir="",
                 raster_dir="",
                 save_image_dir=""
                 ):
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    
    #### NuScenesAgent config ####
    na_config = RawData_NuScenesAgent_config = {
        'version': 'v1.0-mini',
        'debug': False,
        'pred_horizon': 6,
        'obs_horizon': 2,
        'freq': 2,
        'load_dataset': True,
        'debug': False,
        'py_logging_path': None,
        'tb_logging_path': None
    }

    #### NuScenesEnv config ####
    env_config = {
        'NuScenesAgent_config':RawData_NuScenesAgent_config,
        'Sensor_config': {'sensing_patch_size': (60,60), 'agent_road_objects': True},
        'render_paper_ready': True,
        'data_dir': scene_pkl_dir,
        'data_type': 'scene',
        'raster_dir': raster_dir,
        'render_type': ['image'],
        'render_elements': [
            'sim_ego',
            'risk_map',
            'human_ego',
            #'groundtruth',
            #'interaction_labels',
            'token_labels',
            'lanes'
        ],
        'patch_margin': 35,
        'save_image_dir': None,
        'all_info_fields': ['raster_image'],
        'control_mode': 'kinematics'
    }

    env_config = {'type': 'env.dataset_env.NuScenesDatasetEnv', 'config': env_config}

    env = get_env(version=version, 
                  env_path=None, 
                  save_pkl_path=None, 
                  render_bev=True, 
                  config=env_config)

    obs = env.reset(scene_name=scene_name, sample_idx=None)
    done = False
    step = 0
    while not done:
        if obs is None:
            break
        
        action = model.get_action(obs)
        obs, done, other = env.step(action)

        ax = other['render_ax']
        fig = other['render_fig']

        fig.savefig(save_image_dir+"/{:05d}.png".format(step))
        step += 1

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    class DummyModel(object):
        def __init__(self):
            pass
        
        def get_action(self, obs):
            return np.array([0,0])
        
    test_rollout(
        scene_name='scene-0061',
        version='v1.0-mini',
        model=DummyModel(),
        scene_pkl_dir=os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df', 'final'),
        raster_dir=os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df', 'raw', 'scene_df', 'png'),
        save_image_dir=os.getcwd()+'/test'
    )
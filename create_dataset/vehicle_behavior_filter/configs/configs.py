import torch
import os
import copy

import numpy as np

#### NuScenesAgent Config ####
na_config = {'load_dataset': True, 'version': 'v1.0-mini'}

#### Env Config ####
env_config = {
    'NuScenesAgent_config': na_config,
    'Sensor_config': {'agent_road_objects': False, 'sensing_patch_size': (30,30)},
    'render_paper_ready': True,
    'render_type': ['image'],
    #'render_type': [],
    'render_elements': ['sensor_info', 'sim_ego', 'human_ego', 'control_plots'],
    #'render_elements': ['sensor_info', 'sim_ego', 'control_plots'],
    #'render_elements': ['sensor_info'],
    #'patch_margin': 25,
    'patch_margin': 20,
    'save_image_dir': os.path.join(os.environ['PKG_PATH'], 'experiments', 'env_images')
,
    'all_info_fields': ['center_lane'],
    'control_mode': 'kinematics'
}

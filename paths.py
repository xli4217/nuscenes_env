'''
Unified place to store paths 
'''
import os

nuscenes_ds_paths = {
    'local_mini': '/home/xli4217/Xiao/postdoc/TRI/prediction/datasets/NuScenes/data-sample/sets/nuscenes',
    'local_full': '/home/xli4217/Xiao/postdoc/TRI/prediction/datasets/NuScenes/data-full/nuscenes/v1.0',
    'satori_mini': '/nobackup/users/xiaoli/nuscenes_dataset/dataset-mini',
    'satori_full': '/nobackup/users/xiaoli/nuscenes_dataset/dataset',
    'supercloud_mini':'/home/gridsan/xiaoli/data/nuscenes_dataset/dataset-mini',
    'supercloud_full': '/home/gridsan/xiaoli/data/nuscenes_dataset/dataset',

}

mini_path = nuscenes_ds_paths[os.environ['COMPUTE_LOCATION']+'_mini']
full_path = nuscenes_ds_paths[os.environ['COMPUTE_LOCATION']+'_full']
    
experiment_root_dir = os.path.join(os.environ['PKG_PATH'], 'experiments')

experiment_config_dir = os.path.join(os.environ['PKG_PATH'], 'experiment_configs')

processed_data_dir = os.path.join(os.environ['PKG_PATH'], 'create_dataset')

data_analysis_dir = os.path.join(os.environ['PKG_PATH'], 'data_analysis')

scene_img_dir = "/nobackup/users/xiaoli/nuscenes_dataset/scene_imgs"

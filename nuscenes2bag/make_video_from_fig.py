from pathlib import Path
from celluloid import Camera
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def make_video_from_images(image_dir:str=None, image_type='png', video_save_path:str=None, video_layout=None):
    if video_layout is None:
        video_layout = {
            'figsize': (15,15),
            'nb_rows': 6,
            'nb_cols': 6,
            'components': {
                'birdseye': [[0,4], [0,6]],
                'camera': [[4,6], [0,6]]
            },
            'fixed_image': {
                'path': None,
                'layout': [[],[]]
            }

        }

    #img_fn_list = [str(p).split('/')[-1] for p in Path(image_dir).rglob('*.'+image_type)]
    component_img_list = {}
    for k, v in video_layout['components'].items():
        img_fn_list = [str(p).split('/')[-1] for p in Path(image_dir+'/'+k).rglob('*.'+image_type)]
        img_list = [p for p in img_fn_list]
        #img_list = [p for p in img_fn_list if k in p and 'checkpoint' not in p]
        #idx = np.argsort(np.array([int(p[:2]) for p in img_list]))
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
            if i < len(v):
                axes[k].imshow(plt.imread(os.path.join(image_dir,k, v[i])))
        camera.snap()

    animation = camera.animate()

    if video_save_path is not None:
        animation.save(video_save_path)
    return animation

if __name__ == "__main__":
    import os

    experiment_name = 'stl_risk_online_moving_average'
    scene_name = 'scene-0103'
    component_name = 'bev'

    image_dir = "/Users/xiaoli/Xiao/TRI/nuscenes_env/nuscenes2bag/data/supercloud_data/risk_logic_net/final/" + experiment_name + "/"+scene_name+"/"
    print(image_dir)
    video_layout = {
        'figsize': (15,15),
        'nb_rows': 6,
        'nb_cols': 6,
        'components': {
            component_name: [[0,6], [0,6]]
        }
    }

    make_video_from_images(
        image_dir=image_dir, 
        image_type='png', 
        video_save_path=image_dir+'/' + component_name + '/' + component_name+'.mp4', 
        video_layout=video_layout)
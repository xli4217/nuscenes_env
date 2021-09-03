import numpy as np
import torch
import time
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from typing import Tuple, Dict, Callable
import importlib
from prettytable import PrettyTable

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from pathlib import Path
import base64
from celluloid import Camera
import matplotlib.pyplot as plt
import tqdm
import os

#display = Display(visible=0, size=(1400, 900))
#display.start()

def populate_dictionary(d, key, val, val_type, val_shape, populate_func='append'):
    assert_type_and_shape(val, key, val_type, val_shape)
    if populate_func == 'append':
        d[key].append(val)
    elif populate_func == 'assign':
        d[key] = val
    return d

def inspect_processed_dataset_dimensions(df):
    """check to see if all columns have the same shape

    :param df: pandas dataframe
    :returns: None

    """
    tmp = {}
    for idx, row in df.iterrows():
        for k in df.columns.tolist():
            if k not in list(tmp.keys()):
                tmp[k] = [row[k].shape]
            else:
                tmp[k].append(row[k].shape)

    for k, v in tmp.items():
        v = np.array(v)
        print(k, (v ==v[0]).all())

def process_to_len(a, desired_length, name="", dim=0, before_or_after='after', mode='edge', constant_values=0):
    assert a.ndim >= dim, f"need array to have at least {dim} dimensions, right now it has {a.ndim} dimensions"
    #print(name, a.shape, dim)
    dim_list = [(0,0) for _ in range(a.ndim)]
    if before_or_after == 'after':
        dim_list[dim] = (0, desired_length - a.shape[dim])
    else:
        dim_list[dim] = (desired_length - a.shape[dim], 0)
    if a.shape[dim] < desired_length:
        if mode == 'edge':
            a = np.pad(a, dim_list, mode=mode)
        elif mode == 'constant':
            a = np.pad(a, dim_list, mode=mode, constant_values=constant_values)
        else:
            raise ValueError(f'mode {mode} not supported')
    else:
        if dim == 0:
            if before_or_after == 'after':
                a = a[:desired_length]
            elif before_or_after == 'before':
                a = a[-desired_length:]
        elif dim == 1:
            if before_or_after == 'after':
                a = a[:, :desired_length]
            elif before_or_after == 'before':
                a = a[:, -desired_length:]
        else:
            raise ValueError()
    return a


def get_dataframe_summary(d):
    print(f"data shape: {d.shape}")

    d_info = PrettyTable()
    d_info.field_names = ['Column name', 'type', 'shape', 'min', 'mean', 'max']
    for k in d.columns.tolist():
        dk = d.iloc[0][k]
        dk_type = str(type(dk))
        m1 = 'n/a'
        m2 = 'n/a'
        m3 = 'n/a'
        if isinstance(dk, np.ndarray):
            dk_shape = dk.shape
            if len(dk.flatten()) > 0:
                if isinstance(dk.flatten()[0], np.float):
                    m1 = dk.min()
                    m2 = dk.mean()
                    m3 = dk.max()
        elif isinstance(dk, list):
            dk_shape = len(dk)
            # dk = sum(dk, [])
            # if isinstance(dk[0], np.ndarray):
            #     m1 = dk[0].min()
            #     m2 = dk[0].mean()
            #     m3 = dk[0].max()
            # else:
            #     m1 = min(dk)
            #     m2 = 'n/a'
            #     m3 = max(dk)
        else:
            dk_shape = 'n/a'
        
        d_info.add_row([k, dk_type, dk_shape, m1, m2,m3])

    return d_info
#########
# Facet #
#########
from IPython.core.display import display, HTML
import base64
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

def facet_display_overview(data):
    jsonstr = data.to_json(orient='records')
    HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="1000" width="500"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>"""
    html = HTML_TEMPLATE.format(jsonstr=jsonstr)
    display(HTML(html))

def facet_display_stat(data):
    # Create the feature stats for the datasets and stringify it.
    gfsg = GenericFeatureStatisticsGenerator()
    proto = gfsg.ProtoFromDataFrames([{'name': 'train', 'table': data}])
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")

    # Display the facets overview visualization for this data
    HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
        <facets-overview id="elem"></facets-overview>
        <script>
          document.querySelector("#elem").protoInput = "{protostr}";
        </script>"""
    html = HTML_TEMPLATE.format(protostr=protostr)
    display(HTML(html))


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]):
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]


def convert_local_coords_to_global(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]):
    """
    Converts local coordinates to global coordinates.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)

    return np.dot(transform, coordinates.T).T[:, :2] + np.atleast_2d(np.array(translation)[:2])


def show_video_in_jupyter(path):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('utf-8')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))



def make_video_from_images(image_dir:str=None, video_save_dir:str=None, video_layout=None):
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
                if i < len(v):
                    axes[k].imshow(plt.imread(os.path.join(image_dir, v[i])))
            camera.snap()

        animation = camera.animate()

        if video_save_dir is not None:
            animation.save(video_save_dir+'/video.mp4')
        return animation


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

    
class Maxish(torch.nn.Module):
    def __init__(self, name="Maxish input"):
        super(Maxish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1, keepdim=True):
        '''
        The default is
         x is of size [batch_size, N, ...] where N is typically the trace length
         if scale <= 0, then the true max is used, otherwise, the softmax is used.
        '''
        if scale > 0:
            return (torch.softmax(x*scale, dim=dim)*x).sum(dim, keepdim=keepdim)
        else:
            return x.max(dim, keepdim=keepdim)[0]

class Minish(torch.nn.Module):
    def __init__(self, name="Minish input"):
        super(Minish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1, keepdim=True):
        '''
        The default is
         x is of size [batch_size, N, ...] where N is typically the trace length
         if scale <= 0, then the true min is used, otherwise, the softmin is used.
        '''
        if scale > 0:
            return (torch.softmax(-x*scale, dim=dim)*x).sum(dim, keepdim=keepdim)
        else:
            return x.min(dim, keepdim=keepdim)[0]


def assert_type(x, name, expected_type):
    assert isinstance(x, expected_type), name + " is of type {}".format(type(x))


def assert_range(x, name, expected_range):
    pass
    
def assert_shape(x, name, expected_shape):
    if expected_shape is None:
        return True
    assert isinstance(expected_shape, tuple)

    assert len(x.shape) == len(expected_shape), name + f" is of shape {len(x.shape)}, should be {len(expected_shape)}, {name}: {x}"

    for i, xsi, esi in zip(range(len(expected_shape)), x.shape, expected_shape):
        if expected_shape[i] != -1:
            assert xsi == esi, name + " is of shape {}, should be shape {}".format(x.shape, expected_shape)
            
    # if isinstance(x, torch.Tensor):
    #     assert x.size() == expected_shape, name + " is of shape {}, should be shape {}".format(x.size(), expected_shape)
    # elif isinstance(x, np.ndarray):
    #     assert x.shape == expected_shape, name + " is of shape {}, should be shape {}".format(x.shape, expected_shape)
    # elif isinstance(x, tuple):
    #     assert len(x) == expected_shape, name + " is of shape {}, should be shape".format(len(x), expected_shape)
        
def assert_type_and_shape(x, name, expected_type, expected_shape):
    return all([assert_type(x, name, expected_type), assert_shape(x, name, expected_shape)])

def distance_from_point_to_discretized_lane(lane_poses: torch.Tensor, point:torch.Tensor):
    '''
    lane_poses of size (batch_size, nb_lanes, nb_discretized_points, 2)
    point of size (batch_size, 2)
    '''
    batch_size = point.shape[0]
    assert_shape(point, "point", (batch_size, 2))

    # shape (batch_size, nb_lanes * nb_discretized_points, 2)
    lane_poses_flatten = lane_poses.view(lane_poses.shape[0], lane_poses.shape[1]*lane_poses.shape[2], 2)
    distance, idx = torch.norm(point.unsqueeze(1) - lane_poses_flatten, dim=2).min(1)

    closest_pose = lane_poses_flatten[:, idx, :]
    return closest_pose, distance

def batch_distance_from_trajectory_to_discretized_lanes(lane_poses: torch.Tensor, trajectory: torch.Tensor):
    '''
    lane_poses of shape (batch_size, nb_lanes, nb_discretized_points, 2)
    trajectory of shape (batch_size, pred_steps, 2)

    return: distance of shape (batch_size, pred_steps)
    '''

    # shape (batch_size, nb_lanes * nb_discretized_points, 2)
    lane_poses_flatten = lane_poses.view(lane_poses.shape[0], lane_poses.shape[1]*lane_poses.shape[2], 2)
    
    # shape (batch_size, nb_lanes * nb_discretized_points, pred_steps, 2)
    trajectory_expanded = trajectory.unsqueeze(1).repeat(1, lane_poses.shape[1]*lane_poses.shape[2], 1, 1) 

    # shape (batch_size, nb_lanes * nb_discretized_points, pred_steps, 2)
    lane_poses_expanded = lane_poses_flatten.unsqueeze(2).repeat(1, 1, trajectory.shape[1], 1) 

    # shape (batch_size, nb_lanes * nb_discretized_points, pred_steps)
    diff_norm = torch.norm(trajectory_expanded - lane_poses_expanded, dim=3)

    # shape (batch_size, pred_steps)
    #distance, idx = diff_norm.min(1)

    #closest_pose = lane_poses_flatten[torch.arange(lane_poses_flatten.size(0)), idx]

    #return closest_pose, distance

    distance, idx = torch.min(diff_norm, dim=1)

    return None, distance
    
def distance_from_point_to_quadratic_curve(coeff:torch.Tensor, point:torch.Tensor):
    '''
    y = coeff[0]*x**2 + coeff[1]*x + coeff[0]

    coeff has size (batch_size, 3)
    p has size (batch_size, 2)

    '''
    assert_shape(coeff, "coeff", (coeff.shape[0], 3))
    assert_shape(point, "point", (coeff.shape[0], 2))

    # https://archive.lib.msu.edu/crcmath/math/math/p/p406.htm

    a2 = coeff[:,0]
    a1 = coeff[:,1]
    a0 = coeff[:,2]

    x0 = point[:, 0]
    y0 = point[:, 1]

    # a*x^3 + b*x^2 + c*x * d = 0
    a = 2*a2**2
    b = 3*a2*a1
    c = a1**2 + 2*a0*a2 - 2*a2*y0 + 1
    d = a0*a1 - a1*y0 - x0
   
    # https://math.vanderbilt.edu/schectex/courses/cubic/
    p = -b/(3*a)
    q = p.pow(3) + (b*c - 3*a*d)/(6*a.pow(2))
    r = c / (3*a)

    # closest point on quadratic curve
    x11 = ( q.pow(2) + (r-p.pow(2)).pow(3) )
    x1 = ( q + torch.sign(x11) * torch.abs(x11).pow(0.5) )
    x21 = ( q.pow(2) + ( r - p.pow(2) ).pow(3) )
    x2 = ( q - torch.sign(x21)*torch.abs(x21).pow(0.5) ) 
    x = torch.sign(x1)*torch.abs(x1).pow(0.3) + \
        torch.sign(x2)*torch.abs(x2).pow(0.3) + \
        p
    y = a2*x**2 + a1*x + a0

    # distance squared
    d_squared = (x - x0)**2 + (y - y0)**2

    return torch.cat((x.unsqueeze(1), y.unsqueeze(1)), 1), d_squared
    
def split_list_for_multi_worker(input_list:list, worker_num:int):
    if worker_num > len(input_list):
        raise ValueError('worker_num > list length')

    modulo = len(input_list) % worker_num
    if modulo == 0:
        interval = int((len(input_list)) / (worker_num))
    else:
        interval = int((len(input_list)-modulo) / (worker_num-1))

    worker_list = []
    for i in range(0, len(input_list), interval):
        worker_list.append(input_list[i:i+interval])
            
    return worker_list

def timing_val(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__
    return wrapper
    
def get_distance(predictions:np.ndarray, ground_truth:np.ndarray):
    '''
    predictions size is [batch_size, num_modes, prediction_horizon*freq, 2]
    ground_truth size is [batch_size, prediction_horizon*freq, 2]
    
    returns: mean_distance [batch_size, num_modes]
            final_distance [batch_size, num_modes]
            max_distance [batch_size, num_modes]
    '''

    if isinstance(predictions, np.ndarray):
        mean_func = np.mean
        norm_func = np.linalg.norm
        max_func = np.max
    elif isinstance(predictions, torch.Tensor):
        mean_func = torch.mean
        norm_func = torch.norm
        max_func = torch.max
        
    batch_size = predictions.shape[0]
    num_modes = predictions.shape[1]

    # assert_shape(ground_truth, 'ground truth', (batch_size, predictions.shape[2], 2))
    stacked_ground_truth = np.repeat(ground_truth[:, np.newaxis, :, :], num_modes, axis=1)
    
    assert_shape(stacked_ground_truth, 'stacked ground truth', predictions.shape)
        
    dist = np.linalg.norm((predictions - stacked_ground_truth), axis=-1)

    mean_distances = np.mean(dist, axis=-1)
    final_distances = dist[:, :, -1]
    max_distances = np.max(dist, axis=-1)

    assert_shape(mean_distances, 'ade', (batch_size, num_modes))
    assert_shape(final_distances, 'fde', (batch_size, num_modes))
    assert_shape(max_distances, 'maxdist', (batch_size, num_modes))

    return mean_distances, final_distances, max_distances


def set_function_arguments_decorator(config):
    '''
    priority: (1) function input or initialization (not None)
              (2) config (input or initialization None)
    https://avacariu.me/writing/2017/python-decorators-inspecting-function-arguments
    '''
    def set_arguments(f):    
        sig = inspect.signature(f)
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            bound_arguments = sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
        
            for k, v in bound_arguments.items():
                if v is None and k in config.keys():
                    bound_arguments[k] = config[k]
        
            return f(*args, **bound_arguments)
        return wrapper
    return set_arguments

def set_function_arguments_for_class(Cls):
    '''
    https://www.codementor.io/@sheena/advanced-use-python-decorators-class-function-du107nxsv
    '''

    
def set_function_arguments(argument_dict:dict=None, config: dict={}):
        '''
        priority: (1) function input or initialization (not None)
                  (2) config (input or initialization None)
        '''
        r = ()
        for k, v in argument_dict.items():
            if v is not None:
                r += (v, )
            else:
                if k in config.keys():
                    r += (config[k], )
                else:
                    r += (None, )

        return r

def log_data(data_dict: dict={}, py_logger=None, tb_logger=None):
    '''
        data_dict = {
            'py_logger_signal': {
                'name': <name>
            },
            'tb_logger_signal':{
                'trace': {
                    'trace_name': <>
                },
                'histogram':{
                    'hist_name': <>
                },
                'figure': {
                    'fig_name': <>
                }
            }
        }
    '''

    if 'py_logger_signal' in data_dict.keys():
        for k, v in data_dict['py_logger_signal'].items():
            py_logger.info(k + ": " + v)

    if 'tb_logger_signal' in data_dict.keys():
        d = data_dict['tb_logger_signal']
        if 'trace' in d.keys():
            for k, v in d['trace'].items():
                pass
        if 'histogram' in d.keys():
            for k, v in d['histogram'].items():
                pass
        if 'figure' in d.keys():
            for k, v in d['figure'].items():
                pass

def translate_mesh2D(pos, X, Y):
    return X + pos[0], Y + pos[1]

def rotate_mesh2D(pos, rot_rad, X, Y, frame='current'):
    if frame == 'current':
        X -= pos[0]
        Y -= pos[1]

        Xr = np.cos(rot_rad) * X + np.sin(rot_rad) * Y + pos[0]
        Yr = -np.sin(rot_rad) * X + np.cos(rot_rad) * Y + pos[1]
    elif frame == 'global':
        Xr = np.cos(rot_rad) * X + np.sin(rot_rad) * Y 
        Yr = -np.sin(rot_rad) * X + np.cos(rot_rad) * Y
    else:
        raise ValueError('frame not supported')
  
    return Xr, Yr
    
def transform_mesh2D(pos, rot_rad, X, Y):
    Xr = np.cos(rot_rad) * X + np.sin(rot_rad) * Y + pos[0]
    Yr = -np.sin(rot_rad) * X + np.cos(rot_rad) * Y + pos[1]

    return Xr, Yr


class RayWrapper(object):
    def __init__(self, config={}):
        self.config = {
            'num_workers': 1,
            'data_save_path':""
        }
        self.config.update(config)

        

if __name__ == "__main__":
    a = [1,2,3]
    split_list_for_multi_worker(a, 3)

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

import math
import numpy as np
import os
import random
import rosbag
import rospy


class BitMap:
    
    def __init__(self, dataroot: str, map_name: str, layer_name: str):
        """
        This class is used to render bitmap map layers. Currently these are:
        - semantic_prior: The semantic prior (driveable surface and sidewalks) mask from nuScenes 1.0.
        - basemap: The HD lidar basemap used for localization and as general context.

        :param dataroot: Path of the nuScenes dataset.
        :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown` and
            'boston-seaport'.
        :param layer_name: The type of bitmap map, `semanitc_prior` or `basemap.
        """
        self.dataroot = dataroot
        self.map_name = map_name
        self.layer_name = layer_name

        self.image = self.load_bitmap()

    def load_bitmap(self) -> np.ndarray:
        """
        Load the specified bitmap.
        """
        # Load bitmap.
        if self.layer_name == 'basemap':
            map_path = os.path.join(self.dataroot, 'maps', 'basemap', self.map_name + '.png')
        elif self.layer_name == 'semantic_prior':
            map_hashes = {
                'singapore-onenorth': '53992ee3023e5494b90c316c183be829',
                'singapore-hollandvillage': '37819e65e09e5547b8a3ceaefba56bb2',
                'singapore-queenstown': '93406b464a165eaba6d9de76ca09f5da',
                'boston-seaport': '36092f0b03a857c6a3403e25b4b7aab3'
            }
            map_hash = map_hashes[self.map_name]
            map_path = os.path.join(self.dataroot, 'maps', map_hash + '.png')
        else:
            raise Exception('Error: Invalid bitmap layer: %s' % self.layer_name)

        # Convert to numpy.
        if os.path.exists(map_path):
            image = np.array(Image.open(map_path).convert('L'))
        else:
            raise Exception('Error: Cannot find %s %s! Please make sure that the map is correctly installed.'
                            % (self.layer_name, map_path))

        # Invert semantic prior colors.
        if self.layer_name == 'semantic_prior':
            image = image.max() - image

        return image
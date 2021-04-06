import os
import numpy as np

from agents.nuscenes_agent import NuScenesAgent
from utils.nuscenes_virtual_sensing import NuScenesVirtualSensing

default_config = {
    #### NuScenesAgent config ####
    #### NuScenesVirtualSensing config ####
}

class NaiveCostMap(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = default_config
        self.config.update(config)
        
        super(NaiveCostMap, self).__init__(config=self.config, helper=helper, py_logger=py_logger, tb_logger=tb_logger)

        self.name = "NaiveCostMap"

        ####

        self.sensor = NuScenesVirtualSensing(self.config, helper=self.helper, py_logger=self.py_logger, tb_logger=self.tb_logger)

    def update_all_info(self):
        pass

    def update_agent_costmap(self,
                             Xcm:np.ndarray,
                             Ycm:np.ndarray,
                             Zcm:np.ndarray,
                             agent_center:np.ndarray):
        for i in range(Zcm.shape[0]):
            for j in range(Zcm.shape[1]):
                p = np.array([Xcm[i,j], Ycm[i, j]])
                if np.linalg.norm(p - agent_center) < 5:
                    Zcm[i,j] = - np.linalg.norm(p - agent_center)

        return Zcm
                    
    def get_costmap(self, sample_token: str):
        # sensor_info all in global coord ####
        sensor_info = self.sensor.get_info(sample_token)

        ego_info = sensor_info['ego_info']
        agent_info = sensor_info['agent_info']
        map_info = sensor_info['map_info']

        Xcm, Ycm = sensor_info['sensing_patch']['mesh']
        Zcm = np.zeros(Xcm.shape)

        # place a loss around cars
        for agent in agent_info:
            if agent['type'] == 'car':
                c = np.array(agent['translation'])
                Zcm = self.update_agent_costmap(Xcm, Ycm, Zcm, c)
                
        # costmap provided in global frame, 'transform' used to transform to global frame if needed
        costmap_contour = {
            'X': Xcm,
            'Y': Ycm,
            'Z': Zcm,
            'levels': 4,
            'transform': None
        }

        return costmap_contour, sensor_info
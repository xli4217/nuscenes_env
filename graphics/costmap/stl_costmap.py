import os
import numpy as np

from agents.nuscenes_agent import NuScenesAgent
from utils.nuscenes_virtual_sensing import NuScenesVirtualSensing


default_config = {
    #### NuScenesAgent config ####
    #### NuScenesVirtualSensing config ####
}

class StlCostMap(NuScenesAgent):

    def __init__(self, config={}, helper=None, py_logger=None, tb_logger=None):

        self.config = default_config
        self.config.update(config)
        
        super(StlCostMap, self).__init__(config=self.config, helper=helper, py_logger=py_logger, tb_logger=tb_logger)

        self.name = "StlCostMap"

        ####
        self.sensor = NuScenesVirtualSensing(self.config, helper=self.helper, py_logger=self.py_logger, tb_logger=self.tb_logger)

        
    def update_all_info(self):
        pass

    def update_agent_costmap(self,
                             Xcm:np.ndarray,
                             Ycm:np.ndarray,
                             Zcm:np.ndarray,
                             agent_center:np.ndarray,
                             agent_info: dict,
                             map_info: dict):

        for i in range(Zcm.shape[0]):
            for j in range(Zcm.shape[1]):
                p = np.array([Xcm[i,j], Ycm[i, j]])
                Zcm[i,j] += self.get_agent_stl_risk(p, agent_info, map_info)
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
        traj_dist_dict = {}
        for agent in agent_info:
            if agent['type'] == 'car':
                c = np.array(agent['translation'])
                Zcm = self.update_agent_costmap(Xcm, Ycm, Zcm, c, agent, map_info)
                traj_dict = self.get_mock_traj_dict(agent['past'])
                traj_dist_dict[agent['instance_token']+"_"+sample_token] = traj_dict
                
        # costmap provided in global frame, 'transform' used to transform to global frame if needed
        costmap_contour = {
            'X': Xcm,
            'Y': Ycm,
            'Z': Zcm,
            'levels': 4,
            'transform': None
        }

        return costmap_contour, sensor_info, traj_dist_dict

    def get_mock_traj_dict(self, traj):
        traj_dist = []
        for pos in traj:
            traj_dist.append([pos, np.random.rand()*np.eye(2)])

        return { 'traj_dist': traj_dist, 'frame': 'global'}

        
    def get_agent_stl_risk(self, pos:np.ndarray, agent_info: dict, map_info:dict):
        vel = agent_info['velocity']
        
        vel_risk = 0.5 * vel + 0.1
        c = np.array(agent_info['translation'])
        if np.linalg.norm(pos - c) < 5:
            return - vel_risk * np.linalg.norm(pos - c)
        else:
            return 0

        
    # def get_map_risk(self, ego_info:dict, map_info:dict):
    #     pass
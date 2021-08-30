import numpy as np
import pandas as np
import os


def DatasetControlAgent(object):

    def __init__(self, config={}):
        self.config = {}
        self.config.update(config)

    def set_agent_df(self, df):
        self.agent_df = df

    def step(self):
        pass

    def get_safe_controls(self, ctrl: np.ndarray, neighbor_pos: np.ndarray, ego_speed: float):
        def not_zero(x: float, eps: float = 1e-2) -> float:
            if abs(x) > eps:
                return x
            elif x > 0:
                return eps
            else:
                return -eps
            
        # get ado on the same lane #
        same_lane_neighbor_pos = neighbor_pos[np.abs(neighbor_pos[:,0]) < 1., :]
        if len(same_lane_neighbor_pos) == 0:
            return ctrl
        # get closest ado #
        idx = np.argmin(np.abs(same_lane_neighbor_pos[:,1]))
        nearest_neighbor_pos = same_lane_neighbor_pos[idx]

        print(nearest_neighbor_pos)
        dist = abs(abs(nearest_neighbor_pos[1]) - 4)
        if dist > 6.:
            return ctrl

        #### IDM model ####
        delta = 4.0
        comfort_acc_max = 3.0
        comfort_acc_min = -5.0
        desired_gap = 3
        
        # rear vehicle
        target_speed = 8.
        accel = comfort_acc_max * (1 - np.power(max(ego_speed,0)/target_speed, delta))
        accel = min(comfort_acc_max, accel)
        print('idm accel:', accel)
        print('ego speed:', ego_speed)
        
        if nearest_neighbor_pos[1] > 0:
            # front vehicle
            accel -= comfort_acc_max * np.power(desired_gap/not_zero(d), 2)
            accel = max(comfort_acc_min, accel)

        safe_speed = ctrl[0] + accel * 0.5
        ctrl += np.array([safe_speed, 0])
        return ctrl


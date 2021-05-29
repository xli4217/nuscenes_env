import numpy as np

def is_lead_follow(traj1, traj2):
    '''
    returns true if traj1 is following traj2
    '''
    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"

def is_yielding(traj_1, traj2):
    '''
    returns true if traj1 is yielding to traj2
    '''
    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"
    pass

def is_overtaking(traj1, traj2):
    '''
    returns true if traj1 is overtaking traj2
    '''
    assert traj1.shape == traj2.shape, f"shape mismatch with traj1 {traj1.shape} and traj2 {traj2.shape}"
    pass

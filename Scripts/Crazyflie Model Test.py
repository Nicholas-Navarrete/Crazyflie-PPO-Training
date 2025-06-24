import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
import keyboard
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
import numpy as np

Sim_Freq=10000
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'




initial_pos = np.array([[0,0,2]])
initial_pos.reshape(1,3)
env = make_vec_env(HoverAviary,env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=100, pyb_freq=Sim_Freq, initial_xyzs = initial_pos, drone_model = DroneModel.CF2XINVPEN, gui = True),n_envs=1)
obs = env.reset()

model = PPO.load("PPO_Latest_InvPen",print_system_info=True)

#print(env.ACTION_BUFFER_SIZE)
while not keyboard.is_pressed('esc'):
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, truncated, info = env.step(action)

    if keyboard.is_pressed('r'):
        obs = env.reset()
1

import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import keyboard
import numpy as np
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

Sim_Freq=10000
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


'''
UPDATES: Nicholas Navarrete

YawVelocityReward_5 has the first implementation of reset random position gen
With learning batch_size= 2000, verbose=1,learning_rate=0.00001,n_epochs=10,n_steps=100000,ent_coef =0.4, clip_range=0.2

Random Position generation_1 has the best learning with 
batch_size= 2000, verbose=1,learning_rate=0.0001,n_epochs=10,n_steps=100000,ent_coef =0.4, clip_range=0.2,

Random Position generation_2 has implemented the random x an y target to be zero mean, the means were accidentally 2
Random Position generation_5 Learned to fall and stay near 0 x 0 y

Error_Observations_2 actually Flies!!!

Error_Observations_5 also flies, but is trained for longr with no rewards (see tensor board)

0.0001 LR batch size 2000 nsteps 100000 nepochs 10 clip range 0.2 ent_coef = 0.8

Inv Pen_11 Changes the clipping range from 0.2 to 0.1
Inv Pen 13 normalizes the advantage
Inv Pen 14 reduces number of epochs
Inv Pen 17 changes the learning rate
Inv Pen 18 increases pendulum reward coefficient from 0.3 to 0.4 and makes the action buffer size 0.
Inv Pen 19 changes entropy coefficient to 0.01 from 0.8  PEN  19 WORKS WOOOOOOOOOOOOOO BUT IT DOESNT FLY TO THE X Y COORDINATE
Inv Pen 20 readds the reward for the x y position PEN 20 FLIES!!!!!
Inv Pen 21 increases lr to 0.001 from 0.0006 Thsi also flies 
Inv Pen 26 may have been trained with RPM2
Inv Pen 28 is run with gui on to see the randomness
Added trained model rewards
Inv Pen 30 now has updated joint behavior
Inv Pen 32 has reduced pen ang variance from 25 to 10, maybe less noisy rewards?
'''

training_initial_pos = np.array([[0,0,2]])
training_initial_pos.reshape(1,3)

env = make_vec_env(HoverAviary,env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=100, pyb_freq=Sim_Freq, initial_xyzs = training_initial_pos, drone_model = DroneModel.CF2XINVPEN),n_envs=1)

target_reward = 10000
filename = os.path.join('InvPen', 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")) ##was InvPen

if not os.path.exists(filename):
    os.makedirs(filename+'/')

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=0)
eval_callback = EvalCallback(env,callback_on_new_best=callback_on_best,verbose=1,
best_model_save_path=filename+'/',deterministic=True,render=False, eval_freq=10000)    
  



#### Train the model ####################################### 4465
model = PPO('MlpPolicy', env, batch_size= 2000, verbose=1,learning_rate=0.0006,n_epochs=5,n_steps=100000,ent_coef =0.01, clip_range=0.1 , device="cpu",tensorboard_log="./PPO_tensorboard/", normalize_advantage=True)
model.learn(total_timesteps= 1.5e7,
            callback=eval_callback,
            log_interval=10, progress_bar=True,tb_log_name="Inv Pen")#Was Inv Pen

model.save("PPO_Latest_Invpen")#was PPO_Latest_InvPen

env = HoverAviary(gui=True, ctrl_freq=100, pyb_freq=Sim_Freq, act=ActionType('rpm'),drone_model = DroneModel.CF2XINVPEN, initial_xyzs=training_initial_pos, )
obs,info = env.reset()

input("Training is Done! Press enter to continue")
rewardBuffer=0
rewardBufferAverage=[]
while not keyboard.is_pressed('esc'):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewardBuffer=rewardBuffer+reward
    if terminated or truncated:
        rewardBufferAverage.append(rewardBuffer)
        print("average reward over simulations:   " +str(sum(rewardBufferAverage)/len(rewardBufferAverage)))
        rewardBuffer=0
        obs, info = env.reset()
        print("Target Position: " + str(env.TARGET_POS)+ "\n")
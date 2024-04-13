import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
from assignment3 import DQN

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################


# frames = []
# frames.append(env.render())

# Load the trained model
model = DQN(env.observation_space, env.action_space)  # Instantiate a new DQN object
model.load_state_dict(torch.load("trained_dqn_model.pth"))  # Load the state dictionary into the model
state, info = env.reset()
# Define the function to run the environment with the trained model
def run_environment(env, model):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = torch.argmax(q_values).item()
        next_state, reward, done, _, info = env.step(action)
        total_reward += reward
        state = next_state

    print("Total Reward:", total_reward)

# Run the environment with the trained model
run_environment(env, model)

# Close the environment
env.close()



# state, info = env.reset()
# # frames = []
# # frames.append(env.render())

# for i in range(200):
#     action = env.action_space.sample()
#     state, reward, done, _, info = env.step(action)
#     # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
#     if done:
#         break

# env.close()

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
import torch
import torch.nn as nn
import torch.optim as optim

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
#env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################

# Print out the observation space and action space
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)


# state, info = env.reset()
# frames = []
# frames.append(env.render())

class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space.n)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the training loop
def train_dqn(env, dqn, num_episodes=1000, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    optimizer = torch.optim.Adam(dqn.parameters())
    criterion = nn.MSELoss()

    epsilon = epsilon_start
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0))  # Add an extra dimension for batch
                action = torch.argmax(q_values).item()

            next_state, reward, done, _, info = env.step(action)
            # print("State:", state)
            # print("Action:", action)

            # Compute TD target
            q_values_next = dqn(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))  # Add an extra dimension for batch
            max_q_value_next = torch.max(q_values_next).item()
            td_target = reward + gamma * max_q_value_next

            # Compute TD error
            q_values_current = dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0))  # Add an extra dimension for batch
            td_error = td_target - q_values_current[0][action]  # Accessing the Q-value for the selected action

            # Update the Q-value for the selected action
            q_values_current[0][action] += td_error

            # Backpropagation
            optimizer.zero_grad()
            loss = criterion(q_values_current, q_values_next.detach())  # Detach q_values_next to prevent gradient flow
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

            if done:
                break

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print episode information
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # Create the environment
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)

    # Print out the observation space and action space
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)


    state, info = env.reset()
    print(state)

    # Instantiate the DQN network
    observation_space = env.observation_space
    action_space = env.action_space
    dqn = DQN(observation_space, action_space)
    # Load the pre-trained model
    dqn.load_state_dict(torch.load('trained_dqn_model.pth'))    
    # Print out the network architecture
    print(dqn)

    # Train the DQN agent
    train_dqn(env, dqn)
    # Save the trained model
    torch.save(dqn.state_dict(), 'trained_dqn_model.pth')
    # for i in range(200):
    #     action = env.action_space.sample()
    #     state, reward, done, _, info = env.step(action)
    #     # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
    #     if done:
    #         break

    env.close()

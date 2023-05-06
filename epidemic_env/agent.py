"""Implementation of the agent classes and associated RL algorithms.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from epidemic_env.env import Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class Agent(ABC):
    """Implements acting and learning. (Abstract class, for implementations see DQNAgent and NaiveAgent).

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abstractmethod
    def __init__(self,  env, *args, **kwargs):
        """
        Args:
            env (_type_): the simulation environment
        """
        
    @abstractmethod
    def load_model(self, savepath:str):
        """Loads weights from a file.

        Args:
            savepath (str): path at which weights are saved.
        """
        
    @abstractmethod
    def save_model(self, savepath:str):
        """Saves weights to a specified path

        Args:
            savepath (str): the path
        """
        
    @abstractmethod
    def optimize_model(self)->float:
        """Perform one optimization step.

        Returns:
            float: the loss
        """
    
    @abstractmethod
    def reset():
        """Resets the agent's inner state
        """
        
    @abstractmethod 
    def act(self, obs:torch.Tensor)->Tuple[int, float]:
        """Selects an action based on an observation.

        Args:
            obs (torch.Tensor): an observation

        Returns:
            Tuple[int, float]: the selected action (as an int) and associated Q/V-value as a float
        """

class RussoAgent(Agent):
    def __init__(self,  env:Env,
                # Additionnal parameters to be added here
                ):
        """
        Example agent implementation. Just picks a random action at each time step.
        """
        self.env = env
        self.action_dim
        # count number of successive weeks of confinement
        self.conf_counter = 0
        
    def load_model(self, savepath):
        # This is where one would define the routine for loading a pre-trained model
        pass

    def save_model(self, savepath):
        # This is where one would define the routine for saving the weights for a trained model
        pass

    def optimize_model(self):
        # This is where one would define the optimization step of an RL algorithm
        return 0
    
    def reset(self,):
        # This should be called when the environment is reset
        pass
    
    def act(self, obs):
        # this takes an observation and returns an action
        # the action space can be directly sampled from the env

        # Total infected over all the cities at the end of a week
        total_infected = np.sum([obs.city[c].infected[-1] for c in self.env.dyn.cities])

        # End of the confinement, reset the confinement counter. 
        if self.conf_counter == 4:
            self.conf_counter = 0

        # If country is not yet confined and the total of infected people is higher than 20'000 or if the country is within a confinement
        if ((self.conf_counter == 0) and (total_infected>20000)) or ((self.conf_counter > 0)):

            # return the action "confinement" and increase the week of confinement counter by 1
            self.conf_counter += 1
            return 1
        
        # Else do nothing
        self.conf_counter = 0
        return 0 
    

class FactorizedQ(Agent):
    def __init__(self,  env:Env, memory_size=20000 ,learning_rate = 0.005):
        """
        Example agent implementation. Just picks a random action at each time step.
        """
        self.env = env
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.loss_fn = nn.HuberLoss()
        
    
    def create_model(self):
        obs_dim = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]*self.env.observation_space.shape[2]
        action_dim = self.env.action_space.shape[0]*2
        model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load_model(self, savepath):
        # This is where one would define the routine for loading a pre-trained model
        pass

    def save_model(self, savepath):
        # This is where one would define the routine for saving the weights for a trained model
        pass

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        curr_Q = self.model(states)#.gather(1, actions.unsqueeze(1)).squeeze(1)
        print("currentQ", curr_Q, curr_Q.shape)
        print("actions", actions, actions.shape)

        curr_Q = curr_Q.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_Q = self.target_model(next_states).max(1)[0]
        target_Q = rewards + (1 - dones) * 0.9 * next_Q # 0.9 is the discount factor
        loss = self.loss_fn(curr_Q, target_Q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def reset(self,):
        # This should be called when the environment is reset
        pass
    
    def act(self, obs):
        # this takes an observation and returns 4 actions
        # the action space can be directly sampled from the env
        state = obs[0,:2].flatten()
        if np.random.random() < self.epsilon: # Exploration
            return self.env.action_space.sample()
        else: # Exploitation
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)

                # Reshape action_values to gather actions [[a_conf=TRUE, a_conf=FALSE], ...]
                action_values = torch.reshape(action_values, (4,2))
            
            # Return an array of size 4 with binary choice [conf, isol, hosp, vacc]
            return torch.argmax(action_values, dim = 1).tolist()

    def remember(self, state, action, reward, next_state, done):
        #print('Remember :', (state, action, reward, next_state, done))
        #print('State :', state)
        #print('Action :', action)
        #print('Reward :', reward)
        #print('Next state :', next_state)
        #print('Done :', done)
        self.memory.append((state, action, reward, next_state, done))
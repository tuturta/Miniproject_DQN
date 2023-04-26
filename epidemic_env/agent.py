"""Implementation of the agent classes and associated RL algorithms.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from epidemic_env.env import Env

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
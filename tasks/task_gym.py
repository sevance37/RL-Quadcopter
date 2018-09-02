import numpy as np
import gym

class Task_Gym():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,gym_env):
        """Initialize a Task object.
        Params
        ======
            gym_env: the gym_env that you are trying to solve
        """
        # Simulation
        self.sim = gym_env
        #self.sim._max_episode_steps = 1000

        self.state_size = self.sim.observation_space.shape[0]
        self.state_scale = 1

        self.action_low = self.sim.action_space.low[0]
        self.action_high = self.sim.action_space.high[0]
        self.action_size = self.sim.action_space.shape[0]
        self.action_scale = 1
        
    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        next_state, reward, done, _ = self.sim.step(action) 
     
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.sim.reset()
        
        return state
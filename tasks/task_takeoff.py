import numpy as np
from scipy.stats import truncnorm, norm
from physics_sim import PhysicsSim

class Task_Takeoff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pose: target/goal (x,y,z) position for the agent
        """
        
        # initial state
        self.state_scale = 1
        
        self.init_pose = np.concatenate((truncnorm.rvs(-1,1,0,1./3.,3), truncnorm.rvs(-0.021,0.021,0,0.007,3)))
        self.init_pose[2] += 10
        self.init_velocities = np.array([0.,0.,0.])
        self.init_angle_velocities = np.array([0.,0.,0.])

        self.runtime = runtime
        
        # Simulation
        self.sim = PhysicsSim(self.init_pose, self.init_velocities, self.init_angle_velocities, self.runtime) 
        self.action_repeat = 1

        self.init_state = np.concatenate((self.init_pose,self.init_velocities,self.init_angle_velocities),axis=0)
        self.state_size = self.action_repeat * self.init_state.shape[0]
        
        self.action_low = 0 #-1
        self.action_high = 2*450 #1
        self.action_size = 4

        self.action_scale = 1 #450 # 1/2 max of the action 
        #self.state_scale = 150  # 1/2 size of the state space
        
        # Goal
        self.target_pose = np.array([0.,0.,150.0])

        # The previous position
        self.prev_pose = self.init_pose
        
    def get_reward(self, verbose=False):
        
        alpha_t = 1
        alpha_d = 0.3
        alpha_p = 0. #0.1
        alpha_va = 0.00 #0.001
        
        d_tot = np.linalg.norm(self.sim.init_pose[:3]-self.target_pose,1)
        d_curr = np.linalg.norm(self.sim.pose[:3]-self.target_pose,1)
        d_prev = np.linalg.norm(self.prev_pose[:3]-self.target_pose,1)
        
        reward_t = alpha_t*1
        reward_d = alpha_d*(d_prev - d_curr)
        reward_p = alpha_p*np.tanh((1.1*(1-self.sim.time/self.runtime)*d_tot - d_curr)/d_tot); 
        reward_va = -alpha_va*self.sim.angular_v[0]**2 
        
        reward = reward_t + reward_d + reward_p + reward_va
        if (verbose):
            print(reward,reward_t,reward_d,reward_p,reward_va)
        
        return reward
    
    def get_reward_dist(self, verbose=False):
        
        reward = np.tanh(1 - 0.01*(abs(self.sim.pose[:3] - self.target_pose))).sum()
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.prev_pose = self.sim.pose
            done = self.sim.next_timestep(self.action_scale*(rotor_speeds)) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v),axis=0)/self.state_scale)
        next_state = np.concatenate(pose_all)
        # increase reward if target height is reached
        if (self.sim.pose[2] > self.target_pose[2]):
            done = True
            reward += 50*self.runtime
        # reduce reward if it lands on the ground
        if (self.sim.pose[2] == 0):
            done = True
            reward -= 50*self.runtime
        # reduce reward by distance from target at the end of the simulation
        #if (self.sim.time >= self.runtime):
        #    done = True
        #    reward -= np.linalg.norm(self.sim.pose[:3]-self.target_pose,1)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset() 
        state = np.tile(np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v),axis=0)/self.state_scale,self.action_repeat)

        return state
    

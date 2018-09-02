import numpy as np
from scipy.stats import truncnorm, norm
from physics_sim import PhysicsSim

class Task_Hover():
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
        
        alpha_t = 0 
        alpha_x = 0.01 
        alpha_y = 0.01 
        alpha_z = 0.01
        alpha_vx = 0.01
        alpha_vy = 0.01
        alpha_vz = 0.01
        alpha_va = 0.01
        alpha_vb = 0
        alpha_vg = 0
    
        reward_t = alpha_t 
        
        reward_x = -alpha_x*(self.sim.pose[0] - self.target_pose[0])**2 
        reward_y = -alpha_y*(self.sim.pose[1] - self.target_pose[1])**2 
        reward_z = -alpha_z*(self.sim.pose[2] - self.target_pose[2])**2 
        
        reward_vx = -alpha_vx*self.sim.v[0]**2
        reward_vy = -alpha_vy*self.sim.v[1]**2
        reward_vz = -alpha_vz*self.sim.v[2]**2
        
        reward_va = -alpha_va*self.sim.angular_v[0]**2
        reward_vb = -alpha_vb*self.sim.angular_v[1]**2
        reward_vg = -alpha_vg*self.sim.angular_v[2]**2
        
        reward = reward_t + reward_x + reward_y + reward_z \
                + reward_vx + reward_vy + reward_vz + reward_va + reward_vb + reward_vg
        
        if (verbose):
            print(reward, reward_t, reward_x, reward_y, reward_z, 
                  reward_vx, reward_vy, reward_vz, reward_va, reward_vb)
    
    def get_reward_dist(self):
        
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

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset() 
        state = np.tile(np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v),axis=0)/self.state_scale,self.action_repeat)

        return state
    

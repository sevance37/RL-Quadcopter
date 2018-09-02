import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

#=========================================================================================    
import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

#=========================================================================================
from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, params):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
            params (dict):  Parameters for setting up the actor network 
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        self.model_params = params
        
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Parameters for actor network
        ki = initializers.glorot_uniform();
        kr = None
        if self.model_params["use_l2"] > 0:
            l2 = self.model_params["use_l2"]
            kr = regularizers.l2(l2)
        dropout_rate = self.model_params["dropout_rate"]
        use_bias = True
        use_bn = self.model_params["use_bn"]
        if use_bn:
            use_bias=False 
        act_fn = self.model_params["act_fn"]
        n1 = self.model_params["layer1_n"]
        n2 = self.model_params["layer2_n"]
        
        # the actor network
        fc1 = layers.Dense(units=n1, kernel_initializer=ki, kernel_regularizer=kr, use_bias=use_bias, name='fc1')(states)
        if use_bn:
            fc1 = layers.BatchNormalization()(fc1)
        if (dropout_rate > 0):
            fc1 = layers.Dropout(dropout_rate)(fc1)
        fc1 = layers.Activation(act_fn)(fc1)
        
        fc2 = layers.Dense(units=n2, kernel_initializer=ki, kernel_regularizer=kr, use_bias=use_bias, name='fc2')(fc1)
        if use_bn:
            fc2 = layers.BatchNormalization()(fc2)
        if (dropout_rate > 0):
            fc2 = layers.Dropout(dropout_rate)(fc2)
        fc2 = layers.Activation(act_fn)(fc2)

        # Add final output layer
        ki = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        raw_actions = layers.Dense(units=self.action_size, activation='tanh', 
                               kernel_initializer=ki, kernel_regularizer=kr,
                               use_bias=False, name='raw_actions')(fc2)
        # Scale [-1, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (0.5*(x + 1.0)*self.action_range + self.action_low),
                                name='actions')(raw_actions)
        
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        if (kr is not None):
            loss = loss + l2*K.sum(K.square(self.model.get_layer('fc1').get_weights()[0])) \
                        + l2*K.sum(K.square(self.model.get_layer('fc2').get_weights()[0])) \
                        + l2*K.sum(K.square(self.model.get_layer('raw_actions').get_weights()[0]))

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

#=========================================================================================
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, params):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            params (dict):  Parameters for setting up the critic network 
        """
        self.state_size = state_size
        self.action_size = action_size
        
        self.model_params = params
        
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Parameters for critic network
        ki = initializers.glorot_uniform();
        kr = None
        if self.model_params["use_l2"] > 0:
            l2 = self.model_params["use_l2"]
            kr = regularizers.l2(l2)
        dropout_rate = self.model_params["dropout_rate"]            
        use_bias = True
        use_bn = self.model_params["use_bn"]
        if use_bn:
            use_bias=False 
        act_fn = self.model_params["act_fn"]
        n1 = self.model_params["layer1_n"]
        n2 = self.model_params["layer2_n"]
        
        # The critic network
        s_fc1 = layers.Dense(units=n1, kernel_initializer=ki, kernel_regularizer=kr, use_bias=use_bias, name='s_fc1')(states)
        if use_bn:
            s_fc1 = layers.BatchNormalization()(s_fc1)
        if (dropout_rate > 0):
            s_fc1 = layers.Dropout(dropout_rate)(s_fc1)
        s_fc1 = layers.Activation(act_fn)(s_fc1)
        
        a_fc1 = layers.Dense(units=n1, kernel_initializer=ki, kernel_regularizer=kr, use_bias=use_bias, name='a_fc1')(actions)
        if use_bn:
            a_fc1 = layers.BatchNormalization()(a_fc1)
        if (dropout_rate > 0):
            a_fc1 = layers.Dropout(dropout_rate)(a_fc1)
        a_fc1 = layers.Activation(act_fn)(a_fc1)

        # Combine state and action pathways
        net = layers.Add()([s_fc1, a_fc1])
        net = layers.Dense(units=n2, kernel_initializer=ki, kernel_regularizer=kr, use_bias=use_bias, name='fc2')(net)
        if use_bn:
            net = layers.BatchNormalization()(net)
        if (dropout_rate > 0):
            net = layers.Dropout(dropout_rate)(net)
        net = layers.Activation(act_fn)(net)

        # Add final output layer to prduce action values (Q values)
        ki = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        Q_values = layers.Dense(units=1, kernel_initializer=ki, kernel_regularizer=kr,
                                activation='linear', name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

        
#=========================================================================================
import os

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, actor_params, critic_params, noise_params, ddpg_params):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.actor_params = actor_params
        self.critie_params = critic_params

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 self.actor_params)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                  self.actor_params)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.critie_params)
        self.critic_target = Critic(self.state_size, self.action_size, self.critie_params)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = noise_params["mu"]
        self.exploration_theta = noise_params["theta"]  
        self.exploration_sigma = noise_params["sigma"]  
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = ddpg_params["buffer_size"]
        self.batch_size = ddpg_params["batch_size"]
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = ddpg_params["gamma"]
        self.tau = ddpg_params["tau"];
        
        self.weights_dir = ddpg_params["weights_dir"]
        
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, fit_wt, noise_wt, verbose=False):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        noise = 0
        if (noise_wt > 0):
            noise = self.noise.sample()
        if (verbose):
            print(action, noise, noise_wt*noise, fit_wt*action + noise_wt*noise)
        action = fit_wt*action + noise_wt*noise
        action = np.maximum(self.action_low,np.minimum(action,self.action_high))

        return action 

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def save_model(self):
        wts_dir = self.weights_dir
        self.actor_local.model.save_weights(os.path.join(wts_dir,"actor_local.h5"), overwrite=True)
        self.actor_target.model.save_weights(os.path.join(wts_dir,"actor_target.h5"), overwrite=True)
        self.critic_local.model.save_weights(os.path.join(wts_dir,"critic_local.h5"), overwrite=True)
        self.critic_target.model.save_weights(os.path.join(wts_dir,"critic_target.h5"), overwrite=True)

    def load_model(self):
        wts_dir = self.weights_dir
        self.actor_local.model.load_weights(os.path.join(wts_dir,"actor_local.h5"))
        self.actor_target.model.load_weights(os.path.join(wts_dir,"actor_target.h5"))
        self.critic_local.model.load_weights(os.path.join(wts_dir,"critic_local.h5"))
        self.critic_target.model.load_weights(os.path.join(wts_dir,"critic_target.h5"))

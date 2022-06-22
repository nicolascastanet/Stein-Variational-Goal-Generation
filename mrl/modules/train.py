import mrl
from mrl.utils.misc import AttrDict, flatten_state
import numpy as np
from copy import deepcopy
import time

class StandardTrain(mrl.Module):
  def __init__(self, module_name='train', required_agent_modules = ['env', 'policy', 'optimize']):
    super().__init__(module_name, required_agent_modules, locals=locals())

  def _setup(self):
    assert hasattr(self.config, 'optimize_every')
    self.optimize_every = self.config.optimize_every
    self.env_steps = 0
    self.reset_idxs = []

  def __call__(self, num_steps : int, render=False, dont_optimize=False, dont_train=False):
    """
    Runs num_steps steps in the environment, saves collected experiences,
    and trains at every step
    """
    if not dont_train:
      self.agent.train_mode()
    env = self.env
    state = env.state

    for _ in range(num_steps // env.num_envs):
      
      action = self.policy(state)
      next_state, reward, done, info = env.step(action)

      if self.reset_idxs:
        env.reset(self.reset_idxs)
        for i in self.reset_idxs:
          done[i] = True
          if not 'done_observation' in info[i]:
            if isinstance(next_state, np.ndarray):
              info[i].done_observation = next_state[i]
            else:
              for key in next_state:
                info[i].done_observation = {k: next_state[k][i] for k in next_state}
        next_state = env.state
        self.reset_idxs = []

      state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
      self.process_experience(experience)

      if render:
        time.sleep(0.02)
        env.render()
      
      for _ in range(env.num_envs):
        self.env_steps += 1
        if self.env_steps % self.optimize_every == 0 and not dont_optimize:
          self.optimize()
    
    # If using MEP prioritized replay, fit the density model
    if self.config.prioritized_mode == 'mep':
      self.prioritized_replay.fit_density_model()
      self.prioritized_replay.update_priority()
  
  def reset_next(self, idxs):
    """Resets specified envs on next step"""
    self.reset_idxs = idxs

  def save(self, save_folder):
    self._save_props(['env_steps'], save_folder)

  def load(self, save_folder):
    self._load_props(['env_steps'], save_folder)




class AspTrain(mrl.Module):
  def __init__(self, module_name='train', required_agent_modules = ['env', 'policy_A','policy_B','Alice','Bob', 'optimize'], max_steps=50):
    super().__init__(module_name, required_agent_modules, locals=locals())
    self.max_steps = max_steps

  def _setup(self):
    assert hasattr(self.config, 'optimize_every')
    self.optimize_every = self.config.optimize_every
    self.env_steps = 0
    self.reset_idxs = []
    self.goals_B = None

  def __call__(self, num_ep : int, render=False, dont_optimize=False, dont_train=False):
    """
    Runs num_steps steps in the environment, saves collected experiences,
    and trains at every step
    """
    if not dont_train:
      self.agent.train_mode()
    env_B = self.env
    env_A = self.env_A
    state_dim = self.env.state_dim
    num_envs = self.env.num_envs

    for _ in range(int(num_ep // num_envs)):
      
      self.alice_traj = []
      self.bob_traj = []

      # Alice's turn
      state = env_A.reset()

      dones_A = np.zeros((num_envs,))
      steps_A = np.zeros((num_envs,))
      goal_A = state['observation']  # Goal for Alice is the initial state
      final_states_A = np.zeros((num_envs, state_dim))
      dont_record = np.zeros(num_envs) # Keep tracks on finish env

      while not np.all(dones_A):
        
        state = self.relabel_state(state, goal_A)
        action = self.policy_A(state)

        next_state, reward, done, info = env_A.step(action)
        
        for i, (d, inf) in enumerate(zip(done, info)):
          if dones_A[i]:
            continue
          steps_A[i] += 1
          if d:
            dones_A[i] = 1
            final_states_A[i] = state['observation'][i]
            

        state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
        experience.dont_record = deepcopy(dont_record)
        dont_record = deepcopy(dones_A) # Update dont record traj
        self.alice_traj.append(experience)     

        if render:
          time.sleep(0.02)
          env_A.render()


      # Bob's turn
      state = env_B.reset()

      dones_B = np.zeros((num_envs,))
      steps_B = np.zeros((num_envs,))
      self.goals_B = final_states_A # Goal for Bob are the final states for Alice
      success_B = np.zeros((num_envs,))
      dont_record = np.zeros(num_envs) # Keep tracks on finish env

      while not np.all(dones_B):

        state = self.relabel_state(state, self.goals_B)
        action = self.policy_B(state)
        next_state, reward, done, info = env_B.step(action)
        
        for i, (d, inf) in enumerate(zip(done, info)):
          if dones_B[i]:
            continue
          steps_B[i] += 1
          if d:
            dones_B[i] = 1
            success_B[i] = inf['is_success']

        state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
        experience.dont_record = deepcopy(dont_record)
        dont_record = deepcopy(dones_B) # Update dont recoard traj
        self.bob_traj.append(experience)    

        if render:
          time.sleep(0.02)
          env_B.render()

      
      # Compute Alice's reward based on Bob's experience
      for i, (t_a, t_b) in enumerate(zip(steps_A, steps_B)):
        if not success_B[i]:
          t_b = self.max_steps - t_a
        self.alice_traj[int(t_a)-1].reward[i] = max(0,t_b-t_a)

      
      # Now process experiences
      for exp_a, exp_b in zip(self.alice_traj, self.bob_traj):
        self.Alice.replay_buffer._process_experience(exp_a)
        self.Bob.replay_buffer._process_experience(exp_b)
        self.logger._process_experience(exp_b)
      
      for _ in range(num_envs):
        self.env_steps += 1
        if self.env_steps % self.optimize_every == 0 and not dont_optimize:
          self.optimize()
    
    # If using MEP prioritized replay, fit the density model
    if self.config.prioritized_mode == 'mep':
      self.prioritized_replay.fit_density_model()
      self.prioritized_replay.update_priority()
  
  def reset_next(self, idxs):
    """Resets specified envs on next step"""
    self.reset_idxs = idxs

  def relabel_state(self, state, goals):
    """Should be called by the policy module to relabel states with Alice's goals"""

    return {
        'observation': state['observation'],
        'achieved_goal': state['achieved_goal'],
        'desired_goal': goals
    }

  def save(self, save_folder):
    self._save_props(['env_steps'], save_folder)

  def load(self, save_folder):
    self._load_props(['env_steps'], save_folder)

def debug_vectorized_experience(state, action, next_state, reward, done, info):
  """Gym returns an ambiguous "done" signal. VecEnv doesn't 
  let you fix it until now. See ReturnAndObsWrapper in env.py for where
  these info attributes are coming from."""
  experience = AttrDict(
    state = state,
    action = action,
    reward = reward,
    info = info
  )
  next_copy = deepcopy(next_state) # deepcopy handles dict states

  for idx in np.argwhere(done):
    i = idx[0]
    if isinstance(next_copy, np.ndarray):
      next_copy[i] = info[i].done_observation
    else:
      assert isinstance(next_copy, dict)
      for key in next_copy:
        next_copy[key][i] = info[i].done_observation[key]
  
  experience.next_state = next_copy
  experience.trajectory_over = done
  experience.done = np.array([info[i].terminal_state for i in range(len(done))], dtype=np.float32)
  experience.reset_state = next_state
  experience.dont_record = np.zeros(len(reward)) # Record or not trajectory in replay buffer
  
  return next_state, experience
import mrl
from mrl.utils.misc import AttrDict
import numpy as np

class EpisodicEval(mrl.Module):
  def __init__(self, module_name = 'eval', required_agent_modules = ['eval_env', 'policy']):
    super().__init__(module_name, required_agent_modules, locals=locals())

  def _setup(self):

    # Calculate valid set of goals in env
    env = self.eval_env

    if self.config['other_args']['env'] == 'antmaze':
      ant_env = env.env.dummy_env.env
      self.maze_env = ant_env.maze
      self.structure = self.maze_env.MAZE_STRUCTURE

      self.dist_eval = 1.0



    #import ipdb;ipdb.set_trace()
    if self.config['other_args']['env'] == 'pointmaze':
      maze = env.env.dummy_env.env.maze
      self.locs = np.array(list(env.env.dummy_env.env.maze._locs))
      grid = np.copy(self.locs)
      m = np.zeros(grid.shape)
      h = 0.5 # density
      for x in np.arange(-0.5,0.6,h):
          for y in np.arange(-0.5,0.6,h):
              m[:,0] = x
              m[:,1] = y
              grid = np.unique(np.concatenate((grid,self.locs+m)),axis=0)
      self.grid = grid

      self.dist_eval = 0.15

  
  def __call__(self, num_episodes : int, goal=None, log=True, *unused_args):
    """
    Runs num_steps steps in the environment and returns results.
    Results tracking is done here instead of in process_experience, since 
    experiences aren't "real" experiences; e.g. agent cannot learn from them.  
    """
    self.eval_mode()
    env = self.eval_env
    num_envs = env.num_envs
    
    episode_rewards, episode_steps = [], []
    discounted_episode_rewards = []
    is_successes = []
    record_success = False
    max_steps = self.config['other_args']['env_max_step']
    trajs = np.full((num_episodes, num_envs ,max_steps+1, self.env.goal_dim), np.inf)

    ep = 0
    while len(episode_rewards) < num_episodes:
      state = env.reset()

      trajs[ep,:,0] = state['achieved_goal']
    
      dones = np.zeros((num_envs,))
      steps = np.zeros((num_envs,))
      is_success = np.zeros((num_envs,))
      ep_rewards = [[] for _ in range(num_envs)]

      while not np.all(dones):
        if goal is not None:
          state['desired_goal'] = np.expand_dims(goal, axis=0)
        action = self.policy(state)
        state, reward, dones_, infos = env.step(action)

        for i, (ag, rew, done, info) in enumerate(zip(state['achieved_goal'], reward, dones_, infos)):
          if dones[i]:
            continue
          ep_rewards[i].append(rew)
          steps[i] += 1
          try:
            trajs[ep,i,int(steps[i])] = ag
          except IndexError:
            pass
          
          if goal is not None:
            dist = np.linalg.norm(ag - goal, axis=-1)
            if dist < self.dist_eval:
              dones[i] = 1.
              record_success = True
              is_success[i] = 1

          if done:
            dones[i] = 1. 
            if 'is_success' in info:
              record_success = True
              is_success[i] = info['is_success']
  

      ep+=1

      for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
        if record_success:
          is_successes.append(is_succ)
        episode_rewards.append(sum(ep_reward))
        discounted_episode_rewards.append(discounted_sum(ep_reward, self.config.gamma))
        episode_steps.append(step)
    
    if hasattr(self, 'logger') and log==True:
      if len(is_successes):
        self.logger.add_scalar('Test/Success', np.mean(is_successes))
      self.logger.add_scalar('Test/Episode_rewards', np.mean(episode_rewards))
      self.logger.add_scalar('Test/Discounted_episode_rewards', np.mean(discounted_episode_rewards))
      self.logger.add_scalar('Test/Episode_steps', np.mean(episode_steps))

    return AttrDict({
      'rewards': episode_rewards,
      'steps': episode_steps,
      'is_successes' :is_successes,
      'states' :trajs
    })

    
  def success_coverage(self, num_ep=20):
    
    l_goal,_ = self.grid.shape
    output = np.zeros((l_goal,1))

    for i in range(l_goal):
      if i % 100 == 0:
        print("-----\nTest goal success number :", i)
      output[i] = np.mean(self.__call__(num_episodes=num_ep,goal=self.grid[i]).is_successes)

    return output


def discounted_sum(lst, discount):
  sum = 0
  gamma = 1
  for i in lst:
    sum += gamma*i
    gamma *= discount
  return sum
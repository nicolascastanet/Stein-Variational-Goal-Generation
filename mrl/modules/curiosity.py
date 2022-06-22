"""
Curiosity modules for unsupervised exploration
"""

import mrl
import numpy as np
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.misc import softmax, AttrDict
from mrl.utils.svgd import RBF, SVGD, Posterior, Energy, MultivariateGeneralizedGaussian
from sklearn.neighbors import KernelDensity
from collections import deque
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from test_tensor import tensor_point_maze, tensor_ant_maze
from torch.distributions.beta import Beta


def generate_overshooting_goals(num_proposals, step_amount, direct_overshoots, base_goal):
  base_proposals = np.array([base_goal, base_goal + step_amount])
  if direct_overshoots:
    return base_proposals
  additional_proposals = base_goal[None] + np.random.uniform(
      -1.5, 1.5, (num_proposals - 2, step_amount.shape[0])) * step_amount[None]
  return np.concatenate((base_proposals, additional_proposals), 0)


class AchievedGoalCuriosity(mrl.Module):
  """
    For goal agents only. This module assumes the replay buffer maintains an achieved goal buffer;
    To decide on goals to pursue during exploration, the module samples goals from the achieved goal
    buffer, and chooses the highest scoring (see below) viable (per q-function) goal.  
  """
  def __init__(self, num_sampled_ags=500, max_steps=50, keep_dg_percent=-1e-1, randomize=False, sample=False, sample_bis=False, rand=False, get_last_ags=False, use_qcutoff=True, goal_distribution=False):
    super().__init__('ag_curiosity',
                     required_agent_modules=['env', 'replay_buffer', 'actor', 'critic'],
                     locals=locals())
    self.num_sampled_ags = num_sampled_ags
    self.max_steps = max_steps  #TODO: have this be learned from past trajectories?
    self.keep_dg_percent = keep_dg_percent
    self.randomize = randomize
    self.sample = sample
    self.sample_bis = sample_bis
    self.get_last_ags = get_last_ags
    self.use_qcutoff = use_qcutoff
    self.rand = rand

    # if False : choose the behavior goal from the past achieved states, if True : use a specific goal distribution
    self.goal_distribution = goal_distribution


  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env

    self.n_envs = self.env.num_envs
    self.current_goals = None
    self.replaced_goal = np.zeros((self.env.num_envs, ))

    # setup cutoff
    if self.config.gamma < 1.:
      r = min(self.config.gamma, 0.99)
      self.min_min_cutoff = -(1 - r**(self.max_steps * 0.8)) / (1 - r)
    else:
      self.min_min_cutoff = -self.max_steps * 0.8
    self.min_cutoff = max(self.config.initial_cutoff, self.min_min_cutoff)
    self.cutoff = self.min_cutoff

    # go explore + success accounting
    self.go_explore = np.zeros((self.n_envs, 1), dtype=np.float32)
    self.is_success = np.zeros((self.n_envs, 1), dtype=np.float32)
    self.successes_deque = deque(maxlen=10)  # for dynamic cutoff
    self.successes = []

  def _manage_resets_and_success_behaviors(self, experience, close):
    """ Manage (1) end of trajectory, (2) early resets, (3) go explore and overshot goals """
    reset_idxs, overshooting_idxs, overshooting_proposals = [], [], []

    for i, over in enumerate(experience.trajectory_over):
      if over:  # if over update it
        self.current_goals[i] = experience.reset_state['desired_goal'][i]
        self.replaced_goal[i] = 0.
        if np.random.random() < (self.go_explore[i] * self.config.go_reset_percent):
          reset_idxs.append(i)

      if not over and close[i]:  # if not over and success, modify go_explore; maybe overshoot goal?
        self.is_success[i] += 1.
        self.go_explore[i] += 1.

        if not self.config.get('never_done') and np.random.random() < self.config.overshoot_goal_percent:
          step_amount = experience.next_state['achieved_goal'][i] - experience.state['achieved_goal'][i]
          overshooting_idxs.append(i)
          overshooting_proposals.append(
              generate_overshooting_goals(self.num_sampled_ags, step_amount, self.config.direct_overshoots,
                                          self.current_goals[i]))

    return reset_idxs, overshooting_idxs, np.array(overshooting_proposals)

  def _overshoot_goals(self, experience, overshooting_idxs, overshooting_proposals):
    #score the proposals
    num_proposals = overshooting_proposals.shape[1]
    num_idxs = len(overshooting_idxs)
    states = np.tile(experience.reset_state['observation'][overshooting_idxs, None, :], (1, num_proposals, 1))
    states = np.concatenate((states, overshooting_proposals), -1).reshape(num_proposals * num_idxs, -1)

    bad_q_idxs, q_values = [], None
    if self.use_qcutoff:
      q_values = self.compute_q(states)
      q_values = q_values.reshape(num_idxs, num_proposals)
      bad_q_idxs = q_values < self.cutoff
    goal_values = self.score_goals(overshooting_proposals, AttrDict(q_values=q_values, states=states))

    if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
      dg_scores = self.dg_kde.evaluate_log_density(overshooting_proposals.reshape(num_proposals * num_idxs, -1))
      dg_scores = dg_scores.reshape(num_idxs, num_proposals)
      goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

    goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

    chosen_idx = np.argmin(goal_values, axis=1)
    chosen_idx = np.eye(num_proposals)[chosen_idx]  # shape(sampled_ags) = n_envs x num_proposals
    chosen_ags = np.sum(overshooting_proposals * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

    for idx, goal in zip(overshooting_idxs, chosen_ags):
      self.current_goals[idx] = goal
      self.replaced_goal[idx] = 1.

  def _process_experience(self, experience):
    """Curiosity module updates the desired goal depending on experience.trajectory_over"""

    if hasattr(self, 'goal_discriminator'):
      self.goal_discriminator.force_eval = True

    ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag

    if self.current_goals is None:
      self.current_goals = experience.reset_state['desired_goal']

    computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], self.current_goals, 
      {'s':experience.state['observation'], 'ns':experience.next_state['observation']})
    close = computed_reward > -0.5

    # First, manage the episode resets & any special behavior that occurs on goal achievement, like go explore / resets / overshooting
    reset_idxs, overshooting_idxs, overshooting_proposals = self._manage_resets_and_success_behaviors(experience, close)

    if reset_idxs:
      self.train.reset_next(reset_idxs)

    if overshooting_idxs and len(ag_buffer):
      self._overshoot_goals(experience, overshooting_idxs, overshooting_proposals)

    # Now consider replacing the current goals with something else:
    if np.any(experience.trajectory_over) and len(ag_buffer):
      if self.goal_distribution == False:
        if self.get_last_ags:
          # Get the last achieved goals idxs
          sample_idxs = np.arange(len(ag_buffer) - self.num_sampled_ags, len(ag_buffer))
          sample_idxs = np.concatenate([sample_idxs for _ in range(self.n_envs)]) # send the same idxs to all envs

        elif hasattr(self, 'prioritized_replay'):
          #sample_idxs = self.prioritized_replay(self.num_sampled_ags * self.n_envs)
          sample_idxs = self.prioritized_replay(self.num_sampled_ags) # same candidate ags for all envs
          sample_idxs = np.concatenate([sample_idxs for _ in range(self.n_envs)])
        else:
          # sample some achieved goals
          sample_idxs = np.random.randint(len(ag_buffer), size=self.num_sampled_ags * self.n_envs)

        sampled_ags = ag_buffer.get_batch(sample_idxs)
        sampled_ags = sampled_ags.reshape(self.n_envs, self.num_sampled_ags, -1)

        # compute the q-values of both the sampled achieved goals and the current goals
        states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
        states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
        states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
        states_cat = np.concatenate((states, states_curr), 0)


        bad_q_idxs, q_values = [], None
        if self.use_qcutoff:
          q_values = self.compute_q(states_cat)
          q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
          q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

          # Set cutoff dynamically by using intrinsic_success_percent
          if len(self.successes_deque) == 10:
            self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
            intrinsic_success_percent = np.mean(self.successes_deque)
            if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
              self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
              self.successes_deque.clear()
            elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
              self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
              self.successes_deque.clear()

          # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
          bad_q_idxs = q_values < self.cutoff
          q_values[bad_q_idxs] *= -1
          min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
          q_values[bad_q_idxs] *= -1

        # score the goals -- lower is better
        goal_values = self.score_goals(sampled_ags, AttrDict(q_values=q_values, states=states))

        if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
          dg_scores = self.dg_kde.evaluate_log_density(sampled_ags.reshape(self.n_envs * self.num_sampled_ags, -1))
          dg_scores = dg_scores.reshape(self.n_envs, self.num_sampled_ags)
          goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

        if q_values is not None:
          goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

        if self.randomize:  # sample proportional to the absolute score
          abs_goal_values = np.abs(goal_values)
          normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
          chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(1)

        elif self.sample: # sample 1 sample from n_envs distrib

          probas = F.softmax(self.beta * goal_values, dim=1)

          try:
            distrib = Categorical(probas)
          except ValueError:
            import ipdb;ipdb.set_trace()
          chosen_idx = distrib.sample().cpu().numpy()

        elif self.sample_bis: # sample n_envs sample from one distrib
          probas = F.softmax(self.beta * goal_values, dim=0)
          try:
            distrib = Categorical(probas)
          except ValueError:
            import ipdb;ipdb.set_trace()

          chosen_idx = distrib.sample((self.n_envs,)).cpu().numpy()


        elif self.rand: # random sample of candidate indices for all envs

          chosen_idx = np.random.choice(goal_values, self.n_envs)

          """
          # prioritize recent goals
          if len(goal_values) == 1:
            chosen_idx = np.full(self.n_envs, goal_values)
          else:
            idxs = sample_idxs[goal_values]
            idxs = (idxs - min(idxs))/(max(idxs)-min(idxs))
            probas = F.softmax(self.beta * self.torch(idxs), dim=0)
            try:
              distrib = Categorical(probas)
            except ValueError:
              import ipdb;ipdb.set_trace()
            s_idxs = distrib.sample((self.n_envs,)).cpu().numpy()
            chosen_idx = goal_values[s_idxs]"""

        else:  # take minimum
          chosen_idx = np.argmin(goal_values, axis=1)

        chosen_idx = np.eye(self.num_sampled_ags)[chosen_idx]  # shape(sampled_ags) = n_envs x num_sampled_ags
        if q_values is not None:
          chosen_q_val = (chosen_idx * q_values).sum(axis=1, keepdims=True)
        chosen_ags = np.sum(sampled_ags * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

      # If behavioral goals are sampled from another defined distribution
      # score_goals() method directly output the goals instead of ranking previously achieved states
      else:
        q_values = None
        
        chosen_ags = self.score_goals()
        if isinstance(chosen_ags, torch.Tensor):
          chosen_ags = chosen_ags.detach().cpu().numpy()
        # TO DO : reshape for fit in num_envs


      # replace goal always when first_visit_succ (relying on the dg_score_multiplier to dg focus), otherwise
      # we are going to transition into the dgs using the ag_kde_tophat
      if hasattr(self, 'curiosity_alpha'):
        if self.use_qcutoff:
          replace_goal = np.logical_or((np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha),
                                       curr_q < self.cutoff).astype(np.float32)
        else:
          replace_goal = (np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha).astype(np.float32)

      else:
        replace_goal = np.ones((self.n_envs, 1), dtype=np.float32)

      # sometimes keep the desired goal anyways
      replace_goal *= (np.random.uniform(size=[self.n_envs, 1]) > self.keep_dg_percent).astype(np.float32)

      new_goals = replace_goal * chosen_ags + (1 - replace_goal) * self.current_goals

      if hasattr(self, 'logger') and len(self.successes) > 50:
        if q_values is not None:
          self.logger.add_histogram('Explore/Goal_q', replace_goal * chosen_q_val + (1 - replace_goal) * curr_q)
        self.logger.add_scalar('Explore/Intrinsic_success_percent', np.mean(self.successes))
        self.logger.add_scalar('Explore/Cutoff', self.cutoff)
        self.successes = []

      replace_goal = replace_goal.reshape(-1)

      for i in range(self.n_envs):
        if experience.trajectory_over[i]:
          self.successes.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.successes_deque.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.current_goals[i] = new_goals[i]
          if replace_goal[i]:
            self.replaced_goal[i] = 1.
          self.go_explore[i] = 0.
          self.is_success[i] = 0.

    if hasattr(self, 'goal_discriminator'):
      self.goal_discriminator.force_eval = False


  def compute_q(self, numpy_states):
    states = self.torch(numpy_states)
    max_actions = self.actor(states)
    if isinstance(max_actions, tuple):
      max_actions = max_actions[0]
    return self.numpy(self.critic(states, max_actions))

  def relabel_state(self, state):
    """Should be called by the policy module to relabel states with intrinsic goals"""
    if self.current_goals is None:
      return state

    return {
        'observation': state['observation'],
        'achieved_goal': state['achieved_goal'],
        'desired_goal': self.current_goals
    }

  def score_goals(self, sampled_ags, info):
    """ Lower is better """
    raise NotImplementedError  # SUBCLASS THIS!


  def save(self, save_folder):
    self._save_props(['cutoff', 'min_cutoff'], save_folder)  #can restart keeping track of successes / go explore

  def load(self, save_folder):
    self._load_props(['cutoff', 'min_cutoff'], save_folder)


class QAchievedGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals by the Q values (lower is better)
  """
  def score_goals(self, sampled_ags, info):
    scores = np.copy(info.q_values)
    max_score = np.max(scores)
    if max_score > 0:
      scores -= max_score  # so all scores negative

    return scores


class SuccessAchievedGoalCuriosity(AchievedGoalCuriosity):
  """
  fake GOAL GAN
  Scores goals based on success prediction by a goal discriminator module.
  """
  def _setup(self):
    super()._setup()
    self.use_qcutoff = False
    self.mc=False

  def score_goals(self, sampled_ags, info):

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    scores = self.success_predictor(info.states).reshape(num_envs, num_sampled_ags)  # these are predicted success %
    scores = -0.5 + np.abs(scores - 0.5)  # rank by distance to 0.5, lower is closer to 0.5

    return scores

class GoalGAN(AchievedGoalCuriosity):
  def __init__(self, noise_dim, **kwargs):
    super().__init__(**kwargs)
    self.goal_distribution = True
    self.noise_dim = noise_dim
    

  def _setup(self):
    super()._setup()
    self.use_qcutoff = False
    self.mc=False
    self.goal_dim = self.env.goal_dim

  def score_goals(self):

    # GAN warm up
    if self.config.env_steps < 10000 or not self.gan_trainer.ready:
      ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag
      idxs = np.random.randint(len(ag_buffer), size=self.n_envs)
      goals = ag_buffer.get_batch(idxs)

    # GOID from GAN
    else: 
      noise = torch.randn(self.n_envs, self.noise_dim).to(self.config.device)
      goals = self.gan_generator(noise)

    return goals

class SvgdEntropy(AchievedGoalCuriosity):
  def __init__(self, beta = 10, nb_particles=300, epoch=100,annealed=False,slope=1.7,density_module=None,prior_type='fixed_mggd',only_prior=False,proj=False,oe_svgd=20,grad_clip=1000,svgd_lr=1e-1,proj_coord=[4.5,4.5,7],sigma=None,use_prior=False,alpha_beta=None, **kwargs):
    super().__init__(**kwargs)
    self.beta = beta
    self.goal_distribution = True
    self.input_shape = 2 # TO DO : remove hard coded goal shape
    self.n = nb_particles
    self.epoch = epoch
    self.annealed = annealed
    self.slope = slope
    self.density_module = density_module
    self.prior_type = prior_type # in {gm, mggd, learned mggd}
    self.only_prior = only_prior
    self.use_prior = use_prior
    self.proj = proj
    self.optimize_every = oe_svgd
    self.grad_clip = grad_clip
    self.lr = svgd_lr
    self.proj_coord = proj_coord
    self.sigma = sigma
    self.opti_steps = 0

    self.alpha_beta = alpha_beta
    self.annealed_energy = False

  def _setup(self):
    super()._setup()

    self.input_shape = self.env.goal_dim
    self.use_qcutoff = False
    self.mc=False
    self.input_shape = self.env.goal_dim

    if self.config['other_args']['env'] == 'pointmaze':
      self.test_tensor = torch.from_numpy(tensor_point_maze(density=0.1, xy_min=-2.5,xy_max=11.5)).type(torch.float).to(self.config.device)
      self.log_distrib = True
    else:
      self.test_tensor = None
      self.log_distrib = False
    

    if self.density_module is not None:
      self.kde = getattr(self, self.density_module)
    else:
      self.kde = None

    # Prior ditribution
    if self.prior_type == 'fixed_mggd':
        self.prior = MultivariateGeneralizedGaussian(mean=torch.tensor([5,5]).to(self.config.device), input_shape=self.input_shape, alpha=50,beta=4,device=self.config.device)
    elif self.prior_type =='learned_mggd':
        self.prior = getattr(self, 'mggd')
    elif self.prior_type == 'gm':
        self.prior = getattr(self, 'gaussian_mixture')
    elif self.prior_type =='ocsvm':
        self.prior = getattr(self,'OCSVM')
    else:
        self.prior=None
    # RBF kernel and target probability distribution
    
    if self.alpha_beta is None:
      beta = None
    else:
      beta = Beta(torch.tensor(self.alpha_beta[0]).to(self.config.device),torch.tensor(self.alpha_beta[1]).to(self.config.device))

    if self.only_prior:
      goal_disc = None
    else:
      goal_disc = self.goal_discriminator

    self.target = Energy(self.input_shape, goal_disc,self.prior,device=self.config.device,temp=self.beta, use_prior=self.use_prior, ag_kde=self.kde, beta=beta, only_prior=self.only_prior, env=self.eval_env, goal_pred=self.config["other_args"]["goal_predict"])
    self.kernel = RBF(sigma=self.sigma)
    self.fact_prob = 0

    # Init particles and optimizer

    if self.config["other_args"]["goal_predict"]:
      self.init_part = torch.zeros((self.n,self.input_shape))
      for i in range(self.n):
        self.init_part[i] = torch.from_numpy(self.eval_env.reset()["achieved_goal"])

      self.init_part = self.init_part.to(self.config.device)

    else:
      self.init_part = torch.randn(self.n,self.input_shape).to(self.config.device)


    self.particles = self.init_part.clone()
    self.optim = optim.Adam([self.particles], lr=self.lr)


    # SVGD module
    self.svgd = SVGD(self.target, self.kernel, self.optim, self.epoch,device=self.config.device,temp=self.annealed,schedule=1, slope=self.slope)


  def score_goals(self):
    # Update success predictor and ag KDE before update particles
    #self.success_predictor._optimize(force=True)

    if self.prior_type in {'learned_mggd', 'gm'}:
      self.prior.noise = 2
      self.prior.set_noise()
      self.prior.use_noise = True

    #top_k_ind = torch.topk(self.target.log_prob(self.particles),self.n_envs,dim=0)[1]
    #rand_ind = torch.randperm(self.n)[:self.n_envs]
    rand_ind = np.random.choice(self.n,self.n_envs)
    goal_particles = self.particles[rand_ind].squeeze(1)

    #import ipdb;ipdb.set_trace()
    #goal_particles += torch.randn(goal_particles.shape).to(self.config.device)/3
    #self.logger.add_embedding('goal_particles', goal_particles.cpu() ,500, upper_tag='particles')
      
    return goal_particles

  def _optimize(self):
    # Update Particles
    self.opti_steps+=1

    # TO DO : change self.prior.ready
    #if self.opti_steps % 1000 == 0:
    #  print(self.opti_steps)

    
    if self.opti_steps % self.optimize_every == 0 and (not self.use_prior or self.prior.ready) and (self.only_prior or self.success_predictor.is_opt) > 0:
      for t in range(self.epoch):

        if not self.only_prior:
          env_step = self.config.env_steps
          period = self.success_predictor.optimize_every

          # Compute annealed factor      
          if self.annealed:
            if not hasattr(self, 'success_predictor'):
              period = 4000
            a = self.svgd.annealed(env_step,period)
          else:
            a=1

          # Annealed energy (i.e only use prior)
          if self.annealed_energy and (env_step % period) < 1000:
            self.target.annealed_energy = True
        
        else:
          a=1

        # SVGD step
        self.svgd.step(self.particles,t,a)
        if self.particles.isnan().any():
            import ipdb;ipdb.set_trace()

        self.target.annealed_energy = False

        if self.proj:
          c=self.proj_coord[:2]
          d=self.proj_coord[-1]
          self.l2_proj(self.particles,c,d)
          #W+=torch.randn(W.shape)/5
    
      if self.sigma is None:
        self.logger.add_scalar('Particles/sigma_particles', self.kernel.sig_median, log_every=500)

      if self.svgd.norm_cutoff:
        self.logger.add_scalar('Particles/mean_grad_part', float(self.svgd.grad_norm.mean()), log_every=500)
        self.logger.add_scalar('Particles/max_grad_part', float(self.svgd.grad_norm.max()), log_every=500)
      self.logger.add_scalar('Particles/annealed_factor', a, log_every=50)

      self.logger.add_np_embedding('particles', self.particles.cpu() ,500, upper_tag='particles')
      if self.log_distrib:
        self.logger.add_np_embedding('distrib', self.target.log_prob(self.test_tensor, log=False).detach().cpu() ,500, upper_tag='particles')
      

  
  def l2_proj(self,W,c,d):
    """
    y = array, size = (n,2)
    c = [x,y] -> center
    d = float -> radius
    """
    c = torch.repeat_interleave(torch.tensor([c]),W.shape[0],axis=0).to(self.config.device)
    norm = torch.linalg.norm(W-c,axis=1)
    d = torch.repeat_interleave(torch.tensor(d),W.shape[0]).to(self.config.device)

    W-=c
    W*=(d/torch.maximum(norm,d)).reshape(-1,1)
    W+=c

    return W




class LazyRelabelGoals(AchievedGoalCuriosity):
  """
  Failed goals heuristic with policy replay
  """
  def __init__(self, num_ep=2, beta = 100, **kwargs):
    super().__init__(**kwargs)
    self.num_ep = num_ep
    self.beta = beta

  def _setup(self):
    super()._setup()
    self.use_qcutoff = False
    self.mc=False
    assert self.sample_bis != self.rand

  def score_goals(self, sampled_ags, info):

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]
    successes = []

    for i in range(num_sampled_ags):
      res = self.eval(num_episodes=self.num_ep,goal=sampled_ags[0][i],log=False)
      successes.append(res.is_successes)
    self.train_mode() # since eval put in eval mode

    ind_max_ent = np.where(np.mean(successes,1) == 0.5)[0]
    ind_failed = np.where(np.mean(successes,1) == 0)[0]

    if self.sample_bis == True:
      
      scores = self.torch(np.mean(successes,1))
      entropy = -scores * torch.log(scores) -(1-scores)*torch.log(1-scores)
      return torch.nan_to_num(entropy)

    if len(ind_max_ent) != 0:
      return ind_max_ent
    elif len(ind_failed) != 0:
      return ind_failed
    else:
      return np.arange(num_sampled_ags)

 

class SuccessAchievedGoalCuriositySample(AchievedGoalCuriosity):
  """
  Iterative goal sampling (IGS)
  Scores goals based on success prediction by a goal discriminator module.
  """
  def __init__(self, beta=20, nsample = 100, mc = False, **kwargs):
    super().__init__(**kwargs)
    self.beta = beta # Temperature factor
    self.nsamples = nsample
    self.mc = mc

  def _setup(self):
    super()._setup()
    self.use_qcutoff = False
    assert self.beta is not None

  def score_goals(self, sampled_ags, info):

    nsamples = self.nsamples
    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    if self.mc == True:
      outputs = torch.zeros(nsamples, info.states.shape[0], 1)
      for i in range(nsamples):
          outputs[i] = self.torch(self.success_predictor(info.states))
      scores = outputs.mean(0).squeeze().reshape(num_envs, num_sampled_ags)
      scores = -scores * torch.log(scores) -(1-scores)*torch.log(1-scores)
    else:
      scores = self.success_predictor(info.states).reshape(num_envs, num_sampled_ags)  # these are predicted success %
      scores = self.torch(scores)

    # entropy to achieve goals
    entropy = -scores * torch.log(scores) -(1-scores)*torch.log(1-scores)

    return torch.nan_to_num(entropy)


class ThompsonSampling(AchievedGoalCuriosity):
  """
  Iterative goal sampling with epistemic uncertainty w Thompson Sampling
  Scores goals based on success prediction by a goal discriminator module.
  """
  def __init__(self, beta=20, nsample = 100, mc = False, **kwargs):
    super().__init__(**kwargs)
    self.beta = beta # Temperature factor
    self.nsamples = nsample

  def _setup(self):
    super()._setup()
    self.use_qcutoff = False
    assert self.beta is not None

  def score_goals(self, sampled_ags, info):

    nsamples = self.nsamples

    mu = self.success_predictor.mu
    sig = self.success_predictor.sig
    fc_dim = self.success_predictor.last_fc_dim

    posterior = MultivariateNormal(mu, sig)
    weight_samples = posterior.sample((nsamples,))

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    state_dict = self.goal_discriminator.model.state_dict()
    original_weight = state_dict["fc.weight"].detach().clone()

    outputs = torch.zeros(nsamples, info.states.shape[0], 1)
    for i in range(nsamples):
      state_dict["fc.weight"] = weight_samples[i].reshape(1,fc_dim)
      self.goal_discriminator.model.load_state_dict(state_dict)
      outputs[i] = self.torch(self.success_predictor(info.states))

    scores = outputs.mean(0).squeeze().reshape(num_envs, num_sampled_ags)
    entropy = -scores * torch.log(scores) -(1-scores)*torch.log(1-scores)

    state_dict["fc.weight"] = original_weight
    self.goal_discriminator.model.load_state_dict(state_dict)
    
    return torch.nan_to_num(entropy)

class UncertaintyCuriosity(AchievedGoalCuriosity):
  """
  Scores goals based on the uncertainty of the goal discriminator module (MC dropout or else ...)
  mode in {'var-ratios', 'entropy', 'mut-inf'}
  """
  def __init__(self, beta=1.0, MC_samples=100, mode='var', **kwargs):
    super().__init__(**kwargs)
    self.beta = beta # Temperature factor
    self.MC_samples = MC_samples
    self.mode = mode

  def _setup(self):
    super()._setup()
    assert self.beta is not None
    self.use_qcutoff = False

  def score_goals(self, sampled_ags, info):

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]
    states = info.states
    
    MC_samples = self.MC_samples

    pred = np.zeros((states.shape[0], MC_samples, 1)) # NUM_SAMPLES (=n_envs x num_s_ags) x NUM_MC_SAMPLES x 1
    #self.success_predictor.training = True ?
    for i in range(MC_samples):
      with torch.no_grad():
        o = self.compute_q(states) # q_values samples
        pred[:,i] = o

    if self.mode == 'var':
      uncertainty = np.var(pred,axis=1)

    scores = self.torch(uncertainty).reshape(num_envs, num_sampled_ags)

    return scores

  def score_goals_classif(self, sampled_ags, info):

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]
    states = info.states
    MC_samples = self.MC_samples

    probas = torch.zeros(states.shape[0], MC_samples, 2) # NUM_SAMPLES (=n_envs x num_s_ags) x NUM_MC_SAMPLES x NUM_CLASSES
    #self.success_predictor.training = True ?
    for i in range(MC_samples):
      with torch.no_grad():
        o = self.success_predictor(states) # probas samples
        probas[:,i] = self.torch(np.stack([o,1-o],axis=1).squeeze())

    # Predictions
    pred_classes = probas.max(dim=2)[1]
    mean_pred = probas.mean(1).max(dim=1, keepdim=True)[1]

    # Histogramm of predicted classes for each state given the MC sampling 
    hist = np.array([np.histogram(pred_classes[i,:], bins=2)[0]  
                              for i in range(pred_classes.shape[0])])

    # Classification Uncertainty measure
    if self.mode == 'var-ratios':
      uncertainty = 1-hist.max(1)/MC_samples
    elif self.mode == 'entropy':
      mean_entropy = -probas.mean(1)*torch.log(probas.mean(1))
      uncertainty = mean_entropy.sum(1)
    elif self.mode == 'mut-inf':
      mean_entropy = -probas.mean(1)*torch.log(probas.mean(1))
      uncertainty = mean_entropy.sum(1) + (probas*torch.log(probas)).sum(2).mean(1)

    scores = self.torch(uncertainty).reshape(num_envs, num_sampled_ags)

    return scores


class DensityAchievedGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals by their densities (lower is better), using KDE to estimate

  Note on bandwidth: it seems bandwith = 0.1 works pretty well with normalized samples (which is
  why we normalize the ags).
  """
  def __init__(self, density_module='ag_kde', interest_module='ag_interest', alpha=-1.0, **kwargs):
    super().__init__(**kwargs)
    self.alpha = alpha
    self.density_module = density_module
    self.interest_module = interest_module

  def _setup(self):
    assert hasattr(self, self.density_module)
    super()._setup()

  def score_goals(self, sampled_ags, info):
    """ Lower is better """
    density_module = getattr(self, self.density_module)
    if not density_module.ready:
      density_module._optimize(force=True)
    interest_module = None
    if hasattr(self, self.interest_module):
      interest_module = getattr(self, self.interest_module)
      if not interest_module.ready:
        interest_module = None

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    # score the sampled_ags to get log densities, and exponentiate to get densities
    flattened_sampled_ags = sampled_ags.reshape(num_envs * num_sampled_ags, -1)
    sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
    if interest_module:
      # Interest is ~(det(feature_transform)), so we subtract it  in order to add ~(det(inverse feature_transform)) for COV.
      sampled_ag_scores -= interest_module.evaluate_log_interest(flattened_sampled_ags)  # add in log interest
    sampled_ag_scores = sampled_ag_scores.reshape(num_envs, num_sampled_ags)  # these are log densities

    # Take softmax of the alpha * log density.
    # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
    # If alpha < -1, this skews the density to give us low density samples
    normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha)
    normalized_inverse_densities *= -1.  # make negative / reverse order so that lower is better.

    return normalized_inverse_densities


class EntropyGainScoringGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals by their expected entropy gain (higher is better), using KDE to estimate
  current density and another KDE to estimate the joint likelihood of achieved goal 
  given behavioural goal.
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def _setup(self):
    assert hasattr(self, 'bg_kde')
    assert hasattr(self, 'ag_kde')
    assert hasattr(self, 'bgag_kde')
    assert hasattr(self, 'replay_buffer')
    super()._setup()

  def score_goals(self, sampled_ags, info):
    """ Higher entropy gain is better """
    if not self.ag_kde.ready:
      self.ag_kde._optimize(force=True)

    if not self.bg_kde.ready:
      self.bg_kde._optimize(force=True)

    if not self.bgag_kde.ready:
      self.bgag_kde._optimize(force=True)

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    # Get sample of predicted achieved goal from mixture density network
    candidate_bgs = sampled_ags.reshape(num_envs * num_sampled_ags, -1)

    # Reuse the candidate bgs as potential ags
    # Note: We are using a sliding window to reuse sampled_ags as the potential ag for each bg
    # Prior that each bgs has one ag that is identical to bg, i.e. that it reaches the bg.
    num_ags = 10  # TODO: Not make it hard coded
    indexer = np.arange(num_envs * num_sampled_ags).reshape(-1, 1) + np.arange(num_ags).reshape(1, -1)
    indexer %= num_envs * num_sampled_ags  # To wrap around to the beginning
    ags_samples = np.concatenate(
        [candidate_bgs[indexer[i]][np.newaxis, :, :] for i in range(num_envs * num_sampled_ags)], axis=0)

    candidate_bgs_repeat = np.repeat(candidate_bgs[:, np.newaxis, :], num_ags,
                                     axis=1)  # Shape num_envs*num_sampled_ags, num_ags, dim
    joint_candidate_bgags = np.concatenate([candidate_bgs_repeat, ags_samples], axis=-1)
    joint_candidate_bgags = joint_candidate_bgags.reshape(num_envs * num_sampled_ags * num_ags, -1)

    # score the sampled_ags to get log densities, and exponentiate to get densities
    joint_candidate_score = self.bgag_kde.evaluate_log_density(joint_candidate_bgags)
    joint_candidate_score = joint_candidate_score.reshape(num_envs * num_sampled_ags,
                                                          num_ags)  # these are log densities

    candidate_bgs_score = self.bg_kde.evaluate_log_density(
        candidate_bgs_repeat.reshape(num_envs * num_sampled_ags * num_ags, -1))
    candidate_bgs_score = candidate_bgs_score.reshape(num_envs * num_sampled_ags, num_ags)  # these are log densities
    cond_candidate_score = joint_candidate_score - candidate_bgs_score
    cond_candidate_score = softmax(cond_candidate_score, axis=1)

    # Compute entropy gain for the predicted achieved goal
    beta = 1 / len(self.replay_buffer.buffer)
    sampled_ag_entr_new = self.ag_kde.evaluate_elementwise_entropy(candidate_bgs, beta=beta)
    sampled_ag_entr_old = self.ag_kde.evaluate_elementwise_entropy(candidate_bgs, beta=0.)
    sampled_ag_entr_gain = sampled_ag_entr_new - sampled_ag_entr_old
    sampled_ag_entr_gain /= beta  # Normalize by beta # TODO: Get rid of this part if not necessary
    sampled_ag_entr_gain = np.concatenate(
        [sampled_ag_entr_gain[indexer[i]][np.newaxis, :] for i in range(num_envs * num_sampled_ags)], axis=0)
    sampled_ag_entr_gain *= cond_candidate_score
    sampled_ag_entr_gain = sampled_ag_entr_gain.mean(axis=1)

    scores = sampled_ag_entr_gain.reshape(num_envs, num_sampled_ags)
    scores *= -1.  # make negative / reverse order so that lower is better.

    return scores


class CuriosityAlphaMixtureModule(mrl.Module):
  """
    For curiosity agents; this module approximates alpha = (1 / (1 + KL)) using the ag_kde density estimator for p_ag.
  """
  def __init__(self, optimize_every=100):
    super().__init__('curiosity_alpha',
                     required_agent_modules=['ag_curiosity', 'ag_kde', 'replay_buffer'],
                     locals=locals())
    self.samples = None
    self.bandwidth = None
    self.kernel = None
    self.kde = None
    self.fitted_kde = None
    self._alpha = 0.
    self._beta = -3.
    self.step = 0
    self.optimize_every = optimize_every

  def _setup(self):
    self.samples = self.ag_kde.samples
    self.bandwidth = self.ag_kde.bandwidth
    self.kernel = self.ag_kde.kernel
    self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
    if 'curiosity_beta' in self.config:
      self._beta = self.config.curiosity_beta

  @property
  def alpha(self):
    return self._alpha

  def _optimize(self):
    buffer = self.replay_buffer.buffer.BUFF['buffer_dg']
    self.step += 1

    if self.step % self.optimize_every == 0 and len(buffer):

      # Fit the DG KDE
      num_samples = 1000
      sample_idxs = np.random.randint(len(buffer), size=num_samples)
      kde_samples = buffer.get_batch(sample_idxs)
      kde_samples = (kde_samples - self.ag_kde.kde_sample_mean) / self.ag_kde.kde_sample_std
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now compute alpha
      s = kde_samples
      log_p_dg = self.fitted_kde.score_samples(s)
      log_p_ag = self.ag_kde.fitted_kde.score_samples(s)
      self._alpha = 1. / max((self._beta + np.mean(log_p_dg) - np.mean(log_p_ag)), 1.)

      # Occasionally log the alpha
      self.logger.add_scalar('Explore/curiosity_alpha', self._alpha, log_every=500)
      self.logger.add_tabular('Curiosity_alpha', self._alpha)

  def save(self, save_folder):
    self._save_props(['kde', 'samples', 'bandwidth', 'kernel'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'samples', 'bandwidth', 'kernel'], save_folder)

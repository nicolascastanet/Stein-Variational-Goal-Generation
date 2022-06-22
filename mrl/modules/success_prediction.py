"""
Success Prediction Module
"""

from ipdb.__main__ import set_trace
import mrl
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import grad
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal
import numpy as np
import os
from mrl.replays.online_her_buffer import OnlineHERBuffer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from test_tensor import tensor_point_maze, tensor_ant_maze, tensor_slide
import cProfile
import re
import pstats, io
from pstats import SortKey

class GoalSuccessPredictor(mrl.Module):
  """Predicts success using a learned discriminator"""

  def __init__(self, batch_size = 50, history_length = 200, optimize_every=250, log_every=5000, k_steps=1, goal_pred=False):
    super().__init__(
      'success_predictor',
      required_agent_modules=[
        'env', 'replay_buffer', 'goal_discriminator'
      ],
      locals=locals())
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0
    self.is_opt = 0
    self.k_steps = k_steps
    self.k_batch = True
    self.sampling_mode = 'over' # in {'over', 'random', 'balanced', 'smote', 'under_smote', 'MEP'}
    self.n_batch = int(history_length / batch_size)
    self.goal_pred = goal_pred

  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.n_envs = self.env.num_envs
    self.test_tensor = None

    """ Point Maze Test tensor """
    if self.config['other_args']['env'] == 'pointmaze':
      out = tensor_point_maze(xy_min=-2.5,xy_max=11.5)
      goal_test_tensor = torch.from_numpy(out).type(torch.FloatTensor)
      init_state_tensor = torch.zeros((goal_test_tensor.shape[0], 2))
      self.test_tensor = torch.cat((init_state_tensor.to(self.config.device), goal_test_tensor.to(self.config.device)), 1)
    
    elif self.config['other_args']['env'] == 'antmaze':
      out = tensor_ant_maze()
      goal_test_tensor = torch.from_numpy(out).type(torch.FloatTensor)
      obs_init = self.torch(self.eval_env.reset()["observation"])
      init_state_tensor = obs_init.repeat(goal_test_tensor.shape[0],1)
      self.test_tensor = torch.cat((init_state_tensor.to(self.config.device), goal_test_tensor.to(self.config.device)), 1)
    
    elif self.config['other_args']['env'] == 'slide_obj_obj':
      out = tensor_slide(density=0.1)
      goal_test_tensor = torch.from_numpy(out).type(torch.FloatTensor)
      obs_init = self.torch(self.eval_env.reset()["observation"])
      init_state_tensor = obs_init.repeat(goal_test_tensor.shape[0],1)
      self.test_tensor = torch.cat((init_state_tensor.to(self.config.device), goal_test_tensor.to(self.config.device)), 1)
    

  def _optimize(self, force=False):
    self.opt_steps += 1

    
    if len(self.replay_buffer.buffer.trajectories) > self.batch_size and (self.opt_steps % self.optimize_every == 0 or force):
      self.is_opt+=1

      #pr = cProfile.Profile()
      #pr.enable()
      i=0
      #while True:
      #  i+=1
      #  print("while ",i)
        # sampling mode
      if self.sampling_mode == 'balanced':
        inputs, targets, behav_goals = self.balanced_sampling()

      elif self.sampling_mode == 'over':
        inputs, targets, behav_goals = self.over_sampling()

      elif self.sampling_mode == 'smote':
        inputs, targets, behav_goals = self.smote_sampling()

      elif self.sampling_mode == 'under_smote':
        inputs, targets, behav_goals = self.under_smote_sampling()

      elif self.sampling_mode == 'random':
        inputs, targets, behav_goals = self.random_sampling()

      elif self.sampling_mode == 'MEP':
        inputs, targets, behav_goals = self.mep_sampling()

      else:
        raise NotImplementedError

        #if len(torch.unique(targets)) == 2: # check if there is 2 classes
        #  break



      self.optimize_and_log_0(inputs, targets, behav_goals)


# Sampling Functions 

  def mep_sampling(self):
    """
    Samples random past achieved goals with MEP and relabel them by replaying current policy
    """
    k_n = 3
    ratio_obj = 0.5
    
    ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag
    if hasattr(self, 'prioritized_replay'):
      sample_idxs = self.prioritized_replay(self.batch_size)

    sampled_ags = ag_buffer.get_batch(sample_idxs)

    l_goal,_ = sampled_ags.shape
    successes = []

    for i in range(l_goal):
      res = self.eval(num_episodes=1,goal=sampled_ags[i],log=False)
      successes.append(res.is_successes)
    self.train_mode() # since eval put in eval mode

    init_state_tensor = np.zeros((sampled_ags.shape[0], 2))
    inputs = np.concatenate((init_state_tensor, sampled_ags), 1)
    targets = np.array(successes)

    # over sample with SMOTE
    num_succ = targets.sum()
    num_failed = (1-targets).sum()
    min_cl = min(num_succ, num_failed)
    ratio = min_cl/(self.batch_size - min_cl)
    # Check if there is more than 1 class or if objective threshold ratio between classes is already reached
    if (num_succ <= 1) or (len(targets) - num_succ <= 1) or ratio >= 1:
      return self.torch(inputs), self.torch(targets), self.torch(sampled_ags)

    #oversample = SMOTE(sampling_strategy=ratio_obj, k_neighbors=k_n)
    oversample = RandomOverSampler()
    X, y = oversample.fit_resample(inputs, targets)

    targets = self.torch(np.expand_dims(y,1))
    inputs = self.torch(X)
    behav_goals = self.torch(X[:,2:])

    return inputs, targets, behav_goals

  def balanced_sampling(self):
    """
    sample B/2 trajectories over the last H successful / failed ones
    """
    succ_trajs = self.replay_buffer.buffer.sample_succ_trajectories(int(self.batch_size/2), group_by_buffer=True, from_m_most_recent=self.history_length, success=True)
    failed_trajs = self.replay_buffer.buffer.sample_succ_trajectories(int(self.batch_size/2), group_by_buffer=True, from_m_most_recent=self.history_length, success=False)

    successes_0 = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in succ_trajs[2]])
    successes_1 = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in failed_trajs[2]])

    start_states_0 = np.array([t[0] for t in succ_trajs[0]])
    behav_goals_0 =  np.array([t[0] for t in succ_trajs[7]])
    states_0 = np.concatenate((start_states_0, behav_goals_0), -1)

    targets_0 = self.torch(successes_0)
    inputs_0 = self.torch(states_0)


    start_states_1 = np.array([t[0] for t in failed_trajs[0]])
    behav_goals_1 =  np.array([t[0] for t in failed_trajs[7]])
    states_1 = np.concatenate((start_states_1, behav_goals_1), -1)

    targets_1 = self.torch(successes_1)
    inputs_1 = self.torch(states_1)

    targets = torch.cat((targets_0,targets_1))
    inputs = torch.cat((inputs_0,inputs_1))
    behav_goals = torch.cat((self.torch(behav_goals_0),self.torch(behav_goals_1)))

    ind = torch.randperm(targets.shape[0])

    return inputs[ind], targets[ind], behav_goals[ind]

  def over_sampling(self):

    trajs = self.replay_buffer.buffer.sample_trajectories(self.history_length, group_by_buffer=True, from_m_most_recent=self.optimize_every/4)
    successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

    start_states = np.array([t[0] for t in trajs[0]])
    start_goals = np.array([t[0] for t in trajs[6]])
    behav_goals = np.array([t[0] for t in trajs[7]])
    
    if self.goal_pred:
      inputs = start_goals
    else:
      inputs = behav_goals

    states = np.concatenate((inputs, behav_goals), -1)

    # Check if there is more than 1 class
    if successes.sum() == 0 or len(successes) == successes.sum():
      return self.torch(states), self.torch(successes), self.torch(behav_goals)

    # Naive random over sampling
    oversample = RandomOverSampler()
    X, y = oversample.fit_resample(states, successes)

    targets = self.torch(np.expand_dims(y,1))
    inputs = self.torch(X)
    behav_goals = self.torch(X[:,2:])

    return inputs, targets, behav_goals

  def smote_sampling(self):
    k_n = 5
    ratio_obj = 0.5
    trajs = self.replay_buffer.buffer.sample_trajectories(self.history_length, group_by_buffer=True, from_m_most_recent=self.history_length)
    successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

    start_states = np.array([t[0] for t in trajs[0]])
    behav_goals =  np.array([t[0] for t in trajs[7]])
    states = np.concatenate((start_states, behav_goals), -1)

    num_succ = successes.sum()
    num_failed = (1-successes).sum()
    min_cl = min(num_succ, num_failed)
    ratio = min_cl/(self.history_length - min_cl)
    # Check if there is more than 1 class or if objective threshold ratio between classes is already reached
    if (num_succ <= k_n) or (len(successes) - num_succ <= k_n) or ratio >= ratio_obj:
      return self.torch(states), self.torch(successes), self.torch(behav_goals)

    # SMOTE cat (s_0 || g) or just g ?
    oversample = SMOTE(sampling_strategy=ratio_obj, k_neighbors=k_n)
    X, y = oversample.fit_resample(states, successes)

    targets = self.torch(np.expand_dims(y,1))
    inputs = self.torch(X)
    behav_goals = self.torch(X[:,2:])

    return inputs, targets, behav_goals

  def under_smote_sampling(self):
    k_n = 5
    trajs = self.replay_buffer.buffer.sample_trajectories(self.history_length, group_by_buffer=True, from_m_most_recent=self.history_length)
    successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

    start_states = np.array([t[0] for t in trajs[0]])
    behav_goals =  np.array([t[0] for t in trajs[7]])
    states = np.concatenate((start_states, behav_goals), -1)

    # Check if there is more than 1 class
    if (successes.sum() < k_n) or (len(successes) - successes.sum() < k_n):
      return self.torch(successes), self.torch(states), self.torch(behav_goals)

    # SMOTE cat (s_0 || g) or just g ?
    over = SMOTE(sampling_strategy=0.5, k_neighbors=k_n)
    under = RandomUnderSampler()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    X, y = pipeline.fit_resample(states, successes)

    targets = self.torch(y)
    inputs = self.torch(X)
    behav_goals = self.torch(X[:,2:])

    return inputs, targets, behav_goals

  def random_sampling(self):
    trajs = self.replay_buffer.buffer.sample_trajectories(self.history_length, group_by_buffer=True, from_m_most_recent=self.optimize_every)
    successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

    start_states = np.array([t[0] for t in trajs[0]])
    behav_goals =  np.array([t[0] for t in trajs[7]])
    states = np.concatenate((start_states, behav_goals), -1)

    targets = self.torch(successes)
    inputs = self.torch(states)
    behav_goals = self.torch(behav_goals)

    return inputs, targets, behav_goals
    

  def __call__(self, *states_and_maybe_goals):
    """Input / output are numpy arrays"""
    raise NotImplementedError('Subclass this!') 

  def optimize_and_log(self, inputs, targets, behav_goals):
    raise NotImplementedError('Subclass this!')

  def optimize_and_log_0(self, inputs, targets, behav_goals):
    raise NotImplementedError('Subclass this!')  

  def save(self, save_folder : str):
    raise NotImplementedError('Subclass this!') 

  def load(self, save_folder : str):
    raise NotImplementedError('Subclass this!') 



class NNPredictor(GoalSuccessPredictor):
  """
  Use a NN in Pytorch as goals discriminator with gradient steps as optimization
  """
  def _setup(self):
    super()._setup()
    self.optimizer = torch.optim.Adam(self.goal_discriminator.model.parameters())
    if self.test_tensor is not None:
      self.test_tensor = self.test_tensor.to(self.config.device)

    self.compute_post = False

    if self.compute_post:
      # Prior distrib for last fc init
      self.last_fc_dim = self.goal_discriminator.model.fc.weight.squeeze().shape[0]
      self.mu = torch.zeros(self.last_fc_dim).to(self.config.device) # mean
      self.sig = self.torch(np.identity(self.last_fc_dim)) # covariance

  def posterior(self,X,y):
    """
    Laplace approximation: return mean and variance of the posterior that is approximate
    with a multivariate normal distribution 
    """
    try:
      if bool(torch.all(torch.eig(self.sig)[0][:,0]>=0)) == False:
        self.sig+= abs(self.sig.trace())*self.torch(np.identity(self.last_fc_dim))

      last_fc_w = self.goal_discriminator.model.fc.weight
      prior = MultivariateNormal(self.mu, self.sig) # prior distribution
      log_p_w = prior.log_prob(last_fc_w) # log proba of w_map under current prior
    except ValueError:
      import ipdb;ipdb.set_trace()

    self.optimizer.zero_grad()
    y_hat = self.goal_discriminator(X)
    loss = F.binary_cross_entropy_with_logits(y_hat, y) + log_p_w
    gradf_weight = grad(loss, last_fc_w, create_graph=True)[0] # compute gradient wrt to the last fc layer
    
    # Hessian computation
    coeffs = []
    for co in gradf_weight.squeeze():
      grad_c = grad(co, last_fc_w, create_graph=True)[0].detach().cpu().numpy()
      coeffs.append(grad_c)

    hess_weights = np.concatenate(coeffs)
    Sigma_post = np.linalg.inv(hess_weights)

    # Update prior
    self.mu = last_fc_w
    self.sig = self.torch(Sigma_post)


  def optimize_and_log(self, inputs, targets, behav_goals):

    k_behav_goals = []
    k_inputs = []
    k_targets = []

    # k_steps optimization
    for _ in range(self.k_steps):
      
      # Get mini batch
      ind = np.random.randint(targets.shape[0],size=self.batch_size)
      X = inputs[ind]
      y = targets[ind]

      # outputs here have not been passed through sigmoid
      outputs = self.goal_discriminator(X)

      loss = F.binary_cross_entropy_with_logits(outputs, y)

      # optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      k_behav_goals.append(behav_goals[ind])
      k_inputs.append(X)
      k_targets.append(y)

    if self.compute_post:
      self.posterior(torch.cat(k_inputs), torch.cat(k_targets))
    

    if hasattr(self, 'logger'):
      self.goal_discriminator.force_eval = True

      self.logger.add_histogram('predictions', torch.sigmoid(outputs), self.log_every)
      self.logger.add_histogram('targets', targets, self.log_every)

      #accuracy = ((output.squeeze()>=0.5) == y).float().mean()
      #self.logger.add_scalar('Explore/nn_accuracy', np.mean(self.successes))

      if self.test_tensor is not None:
        with torch.no_grad():
          if self.ag_curiosity.mc == True:
            nsamples = self.ag_curiosity.nsamples
            space_pred = torch.zeros(nsamples, self.test_tensor.shape[0], 1)
            goals_pred = torch.zeros(nsamples, torch.cat(k_inputs).shape[0], 1)

            for i in range(nsamples):
              space_pred[i] = torch.sigmoid(self.goal_discriminator(self.test_tensor))
              goals_pred[i] = torch.sigmoid(self.goal_discriminator(torch.cat(k_inputs)))

            space_pred = space_pred.mean(0)
            goals_pred = goals_pred.mean(0)

          else:
            space_pred = torch.sigmoid(self.goal_discriminator(self.test_tensor))
            goals_pred = torch.sigmoid(self.goal_discriminator(inputs))
            accuracy = ((goals_pred.squeeze()>=0.5) == targets.squeeze()).float().mean()

            self.logger.add_scalar('Explore/nn_accuracy', accuracy)


        #self.logger.add_embedding('behav_goals', behav_goals ,self.log_every, upper_tag='success_pred')
        #self.logger.add_embedding('success_labels', targets ,self.log_every, upper_tag='success_pred')
        #self.logger.add_embedding('goals_pred', goals_pred ,self.log_every, upper_tag='success_pred')
        #self.logger.add_embedding('space_pred', space_pred ,self.log_every, upper_tag='success_pred')

        self.goal_discriminator.force_eval = False


  def optimize_and_log_0(self, inputs, targets, behav_goals):

    for _ in range(self.k_steps):

      # Shuffle data at each epoch
      perm = np.random.permutation(self.history_length)
      Xtrain = inputs[perm]
      ytrain = targets[perm]

      for j in range(self.history_length // self.n_batch):

        # Get mini batch
        indsBatch = range(j * self.n_batch, (j+1) * self.n_batch)
        X = Xtrain[indsBatch]
        y = ytrain[indsBatch]

        # outputs here have not been passed through sigmoid
        outputs = self.goal_discriminator(X)

        loss = F.binary_cross_entropy_with_logits(outputs, y)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    if hasattr(self, 'logger'):
      self.goal_discriminator.force_eval = True
      log = False

      if self.test_tensor is not None:
        self.test_tensor.requires_grad = True
        space_pred = torch.sigmoid(self.goal_discriminator(self.test_tensor))
        #grad_x = torch.autograd.grad(space_pred.sum(), self.test_tensor)[0]
        self.logger.add_np_embedding('space_pred', space_pred.detach().cpu() ,self.log_every, upper_tag='success_pred')


      with torch.no_grad():
        goals_pred = torch.sigmoid(self.goal_discriminator(inputs))
        accuracy = ((goals_pred.squeeze()>=0.5) == targets.squeeze()).float().mean()
        self.logger.add_scalar('Explore/nn_accuracy', float(accuracy.cpu()))


      #self.logger.add_embedding('grad_x_norm', torch.linalg.norm(grad_x[:, 2:], dim=1).unsqueeze(1) ,self.log_every, upper_tag='success_pred')
      self.logger.add_np_embedding('behav_goals', behav_goals.cpu() ,self.log_every, upper_tag='success_pred')
      self.logger.add_np_embedding('success_labels', targets.cpu() ,self.log_every, upper_tag='success_pred')
      self.logger.add_np_embedding('goals_pred', goals_pred.cpu() ,self.log_every, upper_tag='success_pred')
      
      self.goal_discriminator.force_eval = False


  def __call__(self, *states_and_maybe_goals):
    """Input / output are numpy arrays"""
    states = np.concatenate(states_and_maybe_goals, -1)
    return self.numpy(torch.sigmoid(self.goal_discriminator(self.torch(states))))

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'opt_state_dict': self.optimizer.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.optimizer.load_state_dict(checkpoint['opt_state_dict'])

class SklearnPredictor(GoalSuccessPredictor):
  """
  Use a SKlearn classifier as goals discriminator : SVM, Knn, Random Forest ...
  """
  def _setup(self):
    super()._setup()


  def optimize_and_log(self, inputs, targets, behav_goals):

    self.goal_discriminator.fit(inputs.cpu(), targets.cpu().squeeze())
    goals_pred = self.goal_discriminator(inputs.cpu())
    
    if hasattr(self, 'logger'):
      self.goal_discriminator.force_eval = True

      self.logger.add_histogram('predictions', goals_pred, self.log_every)
      self.logger.add_histogram('targets', targets, self.log_every)

      if self.test_tensor is not None:
          
        space_pred = self.goal_discriminator(self.test_tensor.cpu())

        self.logger.add_embedding('behav_goals', behav_goals ,self.log_every, upper_tag='success_pred')
        self.logger.add_embedding('success_labels', targets ,self.log_every, upper_tag='success_pred')
        self.logger.add_embedding('goals_pred', np.expand_dims(goals_pred,1) ,self.log_every, upper_tag='success_pred')
        self.logger.add_embedding('space_pred', np.expand_dims(space_pred,1) ,self.log_every, upper_tag='success_pred')

        self.goal_discriminator.force_eval = False


  def __call__(self, *states_and_maybe_goals):
    """Input / output are numpy arrays"""
    states = np.concatenate(states_and_maybe_goals, -1)
    return self.goal_discriminator(states)

  def save(self, save_folder : str):
    pass

  def load(self, save_folder : str):
    pass 
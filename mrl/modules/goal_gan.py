"""
Goal GAN training Module
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
import time

class GoalGanTrainer(mrl.Module):
  """Predicts success using a learned discriminator"""

  def __init__(self, batch_size = 200, history_length = 1000, optimize_every=2000, log_every=5000, k_steps=100, noise_dim=4):
    super().__init__(
      'gan_trainer',
      required_agent_modules=[
        'env', 'replay_buffer', 'goal_discriminator', 'gan_discriminator', 'gan_generator'
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

    self.p_min = 0.1
    self.p_max = 0.9
    self.noise_dim = noise_dim
    self.ready = False
    self.log = True

  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.input_shape = self.env.goal_dim
    self.n_envs = self.env.num_envs
    self.test_tensor = None
    self.optimizerD = torch.optim.Adam(self.gan_discriminator.model.parameters())
    self.optimizerG = torch.optim.Adam(self.gan_generator.model.parameters())
    
    
  def _optimize(self, force=False):
    self.opt_steps += 1
    
    if len(self.replay_buffer.buffer.trajectories) > self.batch_size and (self.opt_steps % self.optimize_every == 0 or force):
      self.is_opt+=1


      i=0
      history = self.history_length
      while True:
        print("loop")
        ##############################################
        ### Over sampling of previous trajectories ###
        ##############################################

        trajs = self.replay_buffer.buffer.sample_trajectories(self.batch_size, group_by_buffer=True, from_m_most_recent=history)
        successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

        start_states = np.array([t[0] for t in trajs[6]])
        behav_goals =  np.array([t[0] for t in trajs[7]])
        states = np.concatenate((start_states, behav_goals), -1)

        ################################
        ### Label GOIDs in real data ###
        ################################

        probas = torch.sigmoid(self.goal_discriminator(self.torch(states))) # are passed through sigmoid
        y_g = (probas > self.p_min) & (probas < self.p_max) # goid or not


        # Check if there is more than 1 class
        if y_g.sum() > 0 and len(y_g) > y_g.sum():
          break
        elif i > 10:
          history = history*2
        elif i > 20:
          break

        i+=1

      # Naive random over sampling
      oversample = RandomOverSampler()
      X, y = oversample.fit_resample(states, y_g.cpu())

      inputs = self.torch(X)
      behav_goals = self.torch(X[:,2:])
      y_g = self.torch(np.expand_dims(y,1))

      
      print("Begin GAN training")
      start = time.time()

      for _ in range(self.k_steps):
        #######################
        ### train  GAN disc ###
        #######################

        L_real_goids = (y_g * (self.gan_discriminator(inputs) - 1)**2).mean()
        L_real_not_goids = (torch.logical_not(y_g) * (self.gan_discriminator(inputs) + 1)**2).mean()

        noise = torch.randn(inputs.shape[0], self.noise_dim).to(self.config.device)
        
        # Sample init state to concat with gen goals
        obs_init = self.torch(self.eval_env.reset()["achieved_goal"]).repeat(inputs.shape[0],1)
        gen_goals = torch.cat((obs_init, self.gan_generator(noise)), 1)
        L_fake = ((self.gan_discriminator(gen_goals) + 1)**2).mean()

        Loss_D = L_real_goids + L_real_not_goids + L_fake

        self.optimizerD.zero_grad()
        Loss_D.backward()
        self.optimizerD.step()

        ############################
        ### train  GAN generator ###
        ############################

        for i in range(2):
          noise = torch.randn(inputs.shape[0], self.noise_dim).to(self.config.device)
          gen_goals = torch.cat((obs_init, self.gan_generator(noise)), 1)
          Loss_G = (self.gan_discriminator(gen_goals)**2).mean()

          self.optimizerG.zero_grad()
          Loss_G.backward()
          self.optimizerG.step()
      
          self.ready = True
        

        if self.log:
          self.logger.add_scalar('Goal_GAN/Gen_Loss', float(Loss_G.detach().cpu()))
          self.logger.add_scalar('Goal_GAN/Disc_Loss', float(Loss_D.detach().cpu()))

      stop = time.time()
      print("End GAN training, time (s) : ",round(stop-start,1))

      

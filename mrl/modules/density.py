"""
Density modules for estimating density of items in the replay buffer (e.g., states / achieved goals).
"""

import mrl
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM
from scipy.special import entr
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.networks import MLP
from mrl.utils.svm import OCSVM
from mrl.utils.svgd import RBF
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution
import os
from mrl.utils.realnvp import RealNVP
import math
#from pykeops.torch import Vi, Vj, LazyTensor
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.nn.functional import softmax, log_softmax

class GenGaussianPytorch(mrl.Module):

    def __init__(self,input_shape=2,beta=6, optimize_every=1000, samples=5000, epoch=200, noise_range=3):
        super().__init__('mggd', required_agent_modules=['replay_buffer'], locals=locals())

        self.input_shape = input_shape
        self.beta = beta
        self.step = 0
        self.item = 'ag'
        self.buffer_name = 'replay_buffer'
        self.optimize_every = optimize_every
        self.normalize = False
        self.samples = samples
        self.epoch = epoch
        self.noise_range = noise_range
        self.use_noise = False

    def _setup(self):
        if self.config.get('device'):
          self.device = self.config.device
        else:
          self.device = 'cpu'

        self.alpha = torch.rand(self.input_shape).type(torch.FloatTensor).to(self.device)
        self.mean = torch.rand(self.input_shape).type(torch.FloatTensor).to(self.device)

        self.alpha.requires_grad, self.mean.requires_grad = (
            True,
            True,
        )

        self.optimizer = torch.optim.Adam([self.mean, self.alpha], lr=0.2)
    
    def log_prob(self, x, beta=None, alpha=None, cov_matr=None):

        if beta is None:
            beta = self.beta
        if alpha is None:
            alpha = self.alpha

        if self.input_shape == 1:
            g = torch.exp((torch.lgamma(torch.tensor(1/beta))))
            norm = beta/(2*alpha*g)
            return norm*torch.exp(-(torch.abs(x-self.mean)/alpha)**beta)

        else:            
            in_s = self.input_shape
            # Norm calcul
            cov = torch.diag(alpha).type(torch.FloatTensor).to(self.device)
            if self.use_noise:
              cov +=self.noise
            #cov = alpha
            if cov_matr is not None:
                cov = cov_matr

            g_1 = torch.exp((torch.lgamma(torch.tensor(in_s/2))))
            g_2 = torch.exp((torch.lgamma(torch.tensor(in_s/(2*beta)))))
            det = torch.det(cov)**(1/2)

            n_1 = g_1/((math.pi**(in_s/2))*g_2*2**(in_s/(2*beta)))
            n_2 = beta/det

            norm = torch.log(n_1*n_2+1e-7)

            # Batch Kernel distance
            bs = x.shape[0]
            x = x.unsqueeze(1)
            mean = self.mean.unsqueeze(0)
            cov = cov.repeat(bs,1,1)
            res_1 = torch.bmm((x-mean),torch.inverse(cov))
            res_2 = torch.bmm(res_1,torch.permute(x-mean, (0, 2, 1)))

            #prob = torch.exp(-1/2*res_2**beta)
            prob = -1/2*res_2**beta

            return (norm + prob).squeeze(1)

    def noisy_cov_log_prob(self,x):
      
      cov = torch.diag(self.alpha)
      noisy_cov = cov + self.noise

      return self.log_prob(x, cov_matr=noisy_cov)


    def set_cov_noise(self):
      cov = torch.diag(self.alpha)
      A = self.noise_range*torch.randn(cov.shape)
      self.noise = torch.mm(A.T,A).to(self.device)



    def _optimize(self):
      # Get batch of achieved goals

      buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
      self.step +=1
      self.use_noise = False

      if (self.step % self.optimize_every == 0 and len(buffer)):

          self.ready = True
          sample_idxs = np.random.randint(len(buffer), size=self.samples)
          gaussian_samples = self.torch(buffer.get_batch(sample_idxs))

          if self.normalize:
              self.kde_sample_mean = np.mean(gaussian_samples, axis=0, keepdims=True)
              self.kde_sample_std  = np.std(gaussian_samples, axis=0, keepdims=True) + 1e-4
              gaussian_samples = self.torch((gaussian_samples - self.kde_sample_mean) / self.kde_sample_std)


          # Optimize with gradient descent on log likelihood
          for _ in range(self.epoch):
              self.optimizer.zero_grad()
              loss = - torch.mean(self.log_prob(gaussian_samples)) 
              loss.backward()
              self.optimizer.step()





class GaussianMixture(mrl.Module):
    def __init__(self, M, item='ag', sparsity=0, D=2, optimize_every=1000, samples=10000, epoch=200):

        super().__init__('gaussian_mixture', required_agent_modules=['replay_buffer'], locals=locals())

        self.M = M; self.D = D
        self.sparsity = sparsity
        self.epoch = epoch
        self.step = 0
        self.optimize_every = optimize_every
        self.samples = samples
        self.buffer_name = 'replay_buffer'
        self.item = item
        self.normalize = False
        self.noise = 0
        self.use_noise = False
        self.noise_batch = None


    def _setup(self):
        if self.config.get('device'):
          self.device = self.config.device
        else:
          self.device = 'cpu'

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.params = {}
        # We initialize our model with random blobs scattered across
        # the unit square, with a small-ish radius:
        self.mu = torch.rand(self.M, self.D).type(dtype).to(self.device)
        self.A = 15 * torch.ones(self.M, 1, 1) * torch.eye(self.D, self.D).view(1,self.D, self.D)
        self.A = (self.A).type(dtype).contiguous().to(self.device)
        self.w = torch.ones(self.M, 1).type(dtype).to(self.device)
        self.mu.requires_grad, self.A.requires_grad, self.w.requires_grad = (
            True,
            True,
            True,
        )

        self.optimizer = torch.optim.Adam([self.A, self.w, self.mu], lr=0.1)
        

    def update_covariances(self):
        """Computes the full covariance matrices from the model's parameters."""
        (M, D, _) = self.A.shape

        AA = torch.matmul(self.A, (self.A).transpose(1, 2))

        # Add noise to covariance
        if self.use_noise:
            cov = torch.stack([torch.inverse(aa) for aa in AA])
            noisy_cov = cov + self.noise_batch
            AA = torch.stack([torch.inverse(c) for c in noisy_cov])

        self.params["gamma"] = (AA).view(
            M, D * D
        ) / 2

    def set_noise(self):
        noise = self.noise * torch.randn(self.A.shape).to(self.device)
        self.noise_batch = torch.matmul(noise,noise.transpose(1,2))
    

    def covariances_determinants(self):
        """Computes the determinants of the covariance matrices.

        N.B.: PyTorch still doesn't support batched determinants, so we have to
              implement this formula by hand.
        """
        S = self.params["gamma"]
        if S.shape[1] == 2 * 2:
            dets = S[:, 0] * S[:, 3] - S[:, 1] * S[:, 2]
        else:
            raise NotImplementedError
        return dets.view(-1, 1)

    def weights(self):
        """Scalar factor in front of the exponential, in the density formula."""
        return softmax(self.w, 0) * self.covariances_determinants().sqrt()

    def weights_log(self):
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) + 0.5 * self.covariances_determinants().log()

    def likelihoods(self, sample):
        """Samples the density on a given point cloud."""
        self.update_covariances()
        return (
            -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))
        ).exp() @ self.weights()

    def log_prob(self, sample):
        """Log-density, sampled on a given point cloud."""
        self.update_covariances()
        K_ij = -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"])) + 1e-7
        return K_ij.logsumexp(dim=1, weight=Vj(self.weights()))

    def neglog_likelihood(self, sample):
        """Returns -log(likelihood(sample)) up to an additive factor."""
        ll = self.log_prob(sample)
        log_likelihood = torch.mean(ll)
        # N.B.: We add a custom sparsity prior, which promotes empty clusters
        #       through a soft, concave penalization on the class weights.
        return -log_likelihood + self.sparsity * softmax(self.w, 0).sqrt().mean()

    def get_sample(self, N):
        """Generates a sample of N points."""
        raise NotImplementedError()

    def _optimize(self):
        # Get batch of achieved goals

        buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
        self.step +=1
        self.use_noise = False

        if (self.step % self.optimize_every == 0 and len(buffer)):

            self.ready = True
            sample_idxs = np.random.randint(len(buffer), size=self.samples)
            gaussian_samples = self.torch(buffer.get_batch(sample_idxs))

            if self.normalize:
                self.kde_sample_mean = np.mean(gaussian_samples, axis=0, keepdims=True)
                self.kde_sample_std  = np.std(gaussian_samples, axis=0, keepdims=True) + 1e-4
                gaussian_samples = self.torch((gaussian_samples - self.kde_sample_mean) / self.kde_sample_std)


            # Optimize with gradient descent on log likelihood
            for _ in range(self.epoch):
                self.optimizer.zero_grad()
                loss = self.neglog_likelihood(gaussian_samples) 
                loss.backward()
                self.optimizer.step()

          
    def plot(self, sample):
        """Displays the model."""
        plt.clf()
        # Heatmap:
        heatmap = self.likelihoods(grid)
        heatmap = (
            heatmap.view(res, res).data.cpu().numpy()
        )  # reshape as a "background" image

        scale = np.amax(np.abs(heatmap[:]))
        plt.imshow(
            -heatmap,
            interpolation="bilinear",
            origin="lower",
            vmin=-scale,
            vmax=scale,
            cmap=cm.RdBu,
            extent=(0, 1, 0, 1),
        )

        # Log-contours:
        log_heatmap = self.log_prob(grid)
        log_heatmap = log_heatmap.view(res, res).data.cpu().numpy()

        scale = np.amax(np.abs(log_heatmap[:]))
        levels = np.linspace(-scale, scale, 41)

        plt.contour(
            log_heatmap,
            origin="lower",
            linewidths=1.0,
            colors="#C8A1A1",
            levels=levels,
            extent=(0, 1, 0, 1),
        )

        # Scatter plot of the dataset:
        xy = sample.data.cpu().numpy()
        plt.scatter(xy[:, 0], xy[:, 1], 100 / len(xy), color="k")



class TorchKDE(mrl.Module):
    """
    Pytorch implementation of KDE for autograd gradient computation
    """
    def __init__(self,item='ag', samples=10000, bw=0.2,buffer_name='replay_buffer',normalize=True):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        super().__init__('torch_kde', required_agent_modules=['replay_buffer'], locals=locals())

        self.samples = samples
        self.item = item
        self.bw = bw
        self.buffer_name = buffer_name
        self.normalize = normalize

    def _setup(self):

        if self.config.get('device'):
          self.device = self.config.device
        else:
          self.device = 'cpu'

        self.dim = self.eval_env.goal_dim
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dim).to(self.device),
                                      covariance_matrix=torch.eye(self.dim).to(self.device))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`.

        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """

        if X == None:
          X = self.get_batch()

        self.n = X.shape[0]

        log_probs = torch.log(
            (self.bw**(-self.dim) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n + 1e-7)

        return log_probs

    def score_samples_1(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.get_batch()

        self.n = X.shape[0]

        X_chunk = X.split(200)
        out = torch.zeros(Y.shape[0]).to(self.device)

        for x in X_chunk:
            probs = (self.bw**(-self.dim) *
                    torch.exp(self.mvn.log_prob(
                    (x.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n

            out+=probs

        return torch.log(out + 1e-7)


    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob


    def get_batch(self):

      buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
      
      self.ready = True
      sample_idxs = np.random.randint(len(buffer), size=self.samples)
      kde_samples = buffer.get_batch(sample_idxs)

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

      return self.torch(kde_samples)


class OCSVMdensity(mrl.Module):
  """
  Density based on One class SVM output
  """
  def __init__(self, item, optimize_every=1000, samples=10000, gamma=0.1,nu=0.01, normalize=True, 
    log_entropy=False, tag='', buffer_name='replay_buffer'):

    super().__init__('OCSVM', required_agent_modules=[buffer_name], locals=locals())

    self.gamma = gamma
    self.nu = nu
    self.step = 0
    self.item = item
    self.optimize_every = optimize_every
    self.samples = samples
    self.normalize = normalize
    self.ready = False
    self.log_entropy = log_entropy
    self.buffer_name = buffer_name

  def _setup(self):
    if self.config.get('device'):
        self.device = self.config.device
    else:
        self.device = 'cpu'

    self.kernel = RBF(gamma=self.gamma)
    self.sk_model = OneClassSVM(nu=self.nu,gamma=self.gamma)
    self.ocsvm = OCSVM(kernel=self.kernel, sk_model=self.sk_model, device=self.device)

    assert isinstance(getattr(self, self.buffer_name), OnlineHERBuffer)

  def _optimize(self, force=False):

    buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
    self.step +=1

    # Add MEP sampling ?
    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      self.ready = True
      if hasattr(self, 'prioritized_replay'):
        sample_idxs = self.prioritized_replay(self.samples)
      else:
        sample_idxs = np.random.randint(len(buffer), size=self.samples)
      ocsvm_samples = buffer.get_batch(sample_idxs)

      #import ipdb;ipdb.set_trace()

      self.ocsvm.fit(ocsvm_samples)
      print("opti ocsvm")
    

  def log_prob(self,x,log=True):
    return self.ocsvm.log_prob(x,log=log)


  def save(self, save_folder):
    self._save_props(['sk_model', 'kernel', 'ocsvm', 'ready'], save_folder)

  def load(self, save_folder):
    self._load_props(['sk_model', 'kernel', 'ocsvm', 'ready'], save_folder)



class RawKernelDensity(mrl.Module):
  """
  A KDE-based density model for raw items in the replay buffer (e.g., states/goals).
  """
  def __init__(self, item, optimize_every=10, samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True, 
    log_entropy=False, tag='', buffer_name='replay_buffer'):

    super().__init__('{}_kde{}'.format(item, tag), required_agent_modules=[buffer_name], locals=locals())

    self.step = 0
    self.item = item
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    self.optimize_every = optimize_every
    self.samples = samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.ready = False
    self.log_entropy = log_entropy
    self.buffer_name = buffer_name

  def _setup(self):
    assert isinstance(getattr(self, self.buffer_name), OnlineHERBuffer)

  def _optimize(self, force=False):
    buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      self.ready = True
      sample_idxs = np.random.randint(len(buffer), size=self.samples)
      kde_samples = buffer.get_batch(sample_idxs)
      #og_kde_samples = kde_samples

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

      #if self.item == 'ag' and hasattr(self, 'ag_interest') and self.ag_interest.ready:
      #  ag_weights = self.ag_interest.evaluate_disinterest(og_kde_samples)
      #  self.fitted_kde = self.kde.fit(kde_samples, sample_weight=ag_weights.flatten())
      #else:
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
      if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
        self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    """ Given an array of samples, compute elementwise function of entropy of the form:

        elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)

    Args:
      samples: 1-D array of size N
      beta: float, offset entropy calculation

    Returns:
      elem_entropy: 1-D array of size N, elementwise entropy with beta offset
    """
    assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
    log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy

  def save(self, save_folder):
    self._save_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)


class RawJointKernelDensity(mrl.Module):
  """
  A KDE-based density model for joint raw items in the replay buffer (e.g., behaviour and achieved goals).

  Args:
    item: a list of items in the replay buffer to build a joint density over
  """
  def __init__(self, items, optimize_every=10, samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True, log_entropy=False, tag=''):
    super().__init__('{}_kde{}'.format("".join(items), tag), required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.items = items
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    self.optimize_every = optimize_every
    self.samples = samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.ready = False
    self.log_entropy = log_entropy

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)

  def _optimize(self, force=False):
    buffers = []
    for item in self.items:
      buffers.append(self.replay_buffer.buffer.BUFF['buffer_' + item])
    
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffers[0])):
      self.ready = True
      sample_idxs = np.random.randint(len(buffers[0]), size=self.samples)
      kde_samples = []
      for buffer in buffers:
        kde_samples.append(buffer.get_batch(sample_idxs))
      
      # Concatenate the items
      kde_samples = np.concatenate(kde_samples, axis=-1)

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std
      
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
      if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).prod()
        self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    """ Given an array of samples, compute elementwise function of entropy of the form:

        elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)

    Args:
      samples: 1-D array of size N
      beta: float, offset entropy calculation

    Returns:
      elem_entropy: 1-D array of size N, elementwise entropy with beta offset
    """
    assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
    log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy

  def save(self, save_folder):
    self._save_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)


class RandomNetworkDensity(mrl.Module):
  """
  A random network based ``density'' model for raw items in the replay buffer (e.g., states/goals). The ``density'' is in proportion
  to the error of the learning network.
  Based on https://arxiv.org/abs/1810.12894.
  """
  def __init__(self, item, optimize_every=1, batch_size=256, layers=(256, 256)):

    super().__init__('{}_rnd'.format(item), required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.item = item
    self.layers = layers
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.tgt_net, self.prd_net, self.optimizer = None, None, None
    self.lazy_load = None

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)

  def _init_from_sample(self, x):
    input_size = x.shape[-1]
    self.tgt_net = MLP(input_size, output_size = self.layers[-1], hidden_sizes = self.layers[:-1])
    self.prd_net = MLP(input_size, output_size = self.layers[-1], hidden_sizes = self.layers[:-1])
    if self.config.get('device'):
      self.tgt_net = self.tgt_net.to(self.config.device)
      self.prd_net = self.prd_net.to(self.config.device)
    self.optimizer = torch.optim.SGD(self.prd_net.parameters(), lr=0.1, weight_decay=1e-5)

  def evaluate_log_density(self, samples):
    """Not actually log density, just prediction error"""
    assert self.tgt_net is not None, "ENSURE READY BEFORE EVALUATING LOG DENSITY"

    samples = self.torch(samples)
    tgt = self.tgt_net(samples)
    prd = self.prd_net(samples)
    return self.numpy(-torch.mean((prd - tgt)**2, dim=-1, keepdim=True))

  @property
  def ready(self):
    return self.tgt_net is not None

  def _optimize(self, force=False):
    buffer = self.replay_buffer.buffer.BUFF['buffer_' + self.item]
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      sample_idxs = np.random.randint(len(buffer), size=self.batch_size)
      samples = buffer.get_batch(sample_idxs)

      # lazy load the networks if not yet loaded
      if self.tgt_net is None:
        self._init_from_sample(samples)
        if self.lazy_load is not None:
          self.load(self.lazy_load)
          self.lazy_load = None

      samples = self.torch(samples)
      tgt = self.tgt_net(samples)
      prd = self.prd_net(samples)
      loss = F.mse_loss(tgt, prd)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.tgt_net is not None:
      torch.save({
        'tgt_state_dict': self.tgt_net.state_dict(),
        'prd_state_dict': self.prd_net.state_dict(),
        'opt_state_dict': self.optimizer.state_dict(),
      }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.tgt_net is None and os.path.exists(path):
      self.lazy_load = save_folder
    else:
      checkpoint = torch.load(path)
      self.tgt_net.load_state_dict(checkpoint['tgt_state_dict'])
      self.prd_net.load_state_dict(checkpoint['prd_state_dict'])
      self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


class FlowDensity(mrl.Module):
  """
  Flow Density model (in this case Real NVP). Similar structure to random density above
  """
  def __init__(self, item, optimize_every=2, batch_size=1000, lr=1e-3, num_layer_pairs=3, normalize=True):

    super().__init__('{}_flow'.format(item), required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.item = item
    self.num_layer_pairs = num_layer_pairs
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.lazy_load = None
    self.flow_model = None
    self.dev = None
    self.lr = lr
    self.sample_mean = 0.
    self.sample_std = 1.
    self.normalize= normalize

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)

  def _init_from_sample(self, x):
    input_size = x.shape[-1]
    self.input_channel = input_size
    if self.config.get('device'):
      self.dev = self.config.device
    elif self.dev is None:
      self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Device=None is fine for default too based on network.py in realNVP
    self.flow_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.dev)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.flow_model.score_samples( (samples - self.sample_mean) / self.sample_std  )

  @property
  def ready(self):
    return self.flow_model is not None

  def _optimize(self, force=False):
    buffer = self.replay_buffer.buffer.BUFF['buffer_' + self.item]
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      sample_idxs = np.random.randint(len(buffer), size=self.batch_size)
      samples = buffer.get_batch(sample_idxs)
      if self.normalize:
        self.sample_mean = np.mean(samples, axis=0, keepdims=True)
        self.sample_std  = np.std(samples, axis=0, keepdims=True) + 1e-4
        samples = (samples - self.sample_mean) / self.sample_std

      # lazy load the model if not yet loaded
      if self.flow_model is None:
        self._init_from_sample(samples)
        if self.lazy_load is not None:
          self.load(self.lazy_load)
          self.lazy_load = None

      samples = self.torch(samples)
      #del self.flow_model
      #self.flow_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.dev)
      self.flow_model.fit(samples, epochs=1)

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.flow_model is not None:
      torch.save({
        'flow_model': self.flow_model,
      }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.flow_model is None and os.path.exists(path):
      self.lazy_load = save_folder
    else:
      self.flow_model = torch.load(path)


"""
class DisagreementDensity(mrl.Module):
  #TODO
"""
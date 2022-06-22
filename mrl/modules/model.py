import mrl
import torch
import numpy as np
from typing import Callable
import os
import pickle
import dill
from joblib import dump, load

class PytorchModel(mrl.Module):
  """
  Generic wrapper for a pytorch nn.Module (e.g., the actorcritic network).
  These live outside of the learning algorithm modules so that they can easily be 
  shared by different modules (e.g., critic can be used by intrinsic curiosity module). 
  They are also saved independently of the agent module (which is stateless). 
  """

  def __init__(self, name : str, model_fn : Callable):
    super().__init__(name, required_agent_modules=[], locals=locals())
    self.model_fn = model_fn
    self.model = self.model_fn()
    self.force_eval = False

  def _setup(self):
    if self.config.get('device'):
      self.model = self.model.to(self.config.device)

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save(self.model.state_dict(), path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    self.model.load_state_dict(torch.load(path), strict=False)

  def copy(self, new_name):
    """Makes a copy of the Model; e.g., for target networks"""
    new_model = dill.loads(dill.dumps(self.model))
    model_fn = lambda: new_model
    return self.__class__(new_name, model_fn)

  def __call__(self, *args, **kwargs):
    if self.training and not self.force_eval:
      self.model.train()
    else:
      self.model.eval()
    return self.model(*args, **kwargs)



class SklearnModel(mrl.Module):
  """
  Generic wrapper for a Sklearn model.
  These live outside of the learning algorithm modules so that they can easily be 
  shared by different modules (e.g., critic can be used by intrinsic curiosity module). 
  They are also saved independently of the agent module (which is stateless). 
  """

  def __init__(self, name : str, model_fn : Callable):
    super().__init__(name, required_agent_modules=[], locals=locals())
    self.model_fn = model_fn
    self.model = self.model_fn()
    self.force_eval = False
    self.fitted = False

  def _setup(self):
    pass

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.joblib')
    dump(self.model, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.joblib')
    self.model = load(path) 
    

  def copy(self, new_name):
    """Makes a copy of the Model; e.g., for target networks"""
    new_model = dill.loads(dill.dumps(self.model))
    model_fn = lambda: new_model
    return self.__class__(new_name, model_fn)

  def fit(self, X, y):
    self.model.fit(X,y)
    self.fitted = True

  def __call__(self, *args, **kwargs):
    if self.fitted == False:
      return np.random.rand(5500) # TO DO : remove hard coded shape for init random pred
    else:
      proba = self.model.predict_proba(*args, **kwargs)
      _,cl = proba.shape
      if cl == 1:
        return proba
      else:
        return proba[:,1]
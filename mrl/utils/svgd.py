"""
Stein Variational Gradient Descent : https://arxiv.org/pdf/1608.04471.pdf
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.distributions as dist
from scipy.spatial import Delaunay
import traceback

# RBF Kernel
class RBF(torch.nn.Module):
    def __init__(self, sigma=None,gamma=None,sig_mult=1):
        super(RBF, self).__init__()

        self.sigma = sigma
        self.gamma = gamma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
            self.sig_median=sigma
        else:
            sigma = self.sigma

        if self.gamma is None:
            gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        else:
            gamma = self.gamma
        K_XY = (-gamma * dnorm2).exp()
  
        return K_XY


# Stein Variational Gradient Descent
class SVGD:
    def __init__(self, P, K, optimizer, epoch,device=None, nn_post=False, temp = False, p=10, schedule=1, slope=1.7):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.temp = temp
        self.T = epoch
        self.p = p
        self.schedule = schedule
        self.slope = slope
        self.nn_post = nn_post
        self.device = 'cpu' if device is None else device
        self.max_norm = 1000
        self.norm_cutoff = False

    def phi(self, X,t,ann=1):
        X = X.detach().requires_grad_(True).to(self.device)

        if self.nn_post:
            _, score_func = self.P.log_prob(X)
        
        else:
            log_prob = self.P.log_prob(X)
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True)[0]

            if score_func.isnan().any():
                #print("NAN !")
                #import ipdb;ipdb.set_trace()
                score_func = torch.nan_to_num(score_func)
                
            if self.norm_cutoff:
                grad_x = autograd.grad(self.P.net_prob.sum(), X)[0]
                norm = torch.linalg.norm(grad_x,axis=1)
                ind = torch.where(norm > self.max_norm)[0]
                if len(ind) > 0:
                    score_func[ind] = 0

                self.grad_norm = norm
        

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]
        if grad_K.isnan().any():
            import ipdb;ipdb.set_trace()

        phi = (K_XX.detach().matmul(score_func)*ann + grad_K) / X.size(0)
        if phi.isnan().any():
            
            phi = torch.nan_to_num(phi)
            #import ipdb;ipdb.set_trace()
            
        return phi

    def step(self, X,t,ann=False):
        self.optim.zero_grad()
        X.grad = -self.phi(X,t,ann)
        if X.grad.isnan().any():
            import ipdb;ipdb.set_trace()
        self.optim.step()


    def annealed(self,t,period,T=1e6,p=5, mode=2, C=4):
        if mode == 1:
            return np.tanh((self.slope*t/T)**p)
        elif mode == 2:
            t = t % period
            return np.tanh((self.slope*t/period)**p)
        elif mode==3:
            return int(t > (T//2))


# Posterior of NN parameters
class Posterior():
    def __init__(self,input_shape, X, y, net, sig_prior=1, b_size=100):
        self.sig_prior = sig_prior
        self.input_shape = input_shape
        self.prior = dist.multivariate_normal.MultivariateNormal(torch.zeros(input_shape), sig_prior*torch.eye(input_shape))
        self.loss = nn.BCELoss()
        self.net = net
        self.X = X
        self.y = y
        self.b_size = b_size
        

    def log_prob(self, value):
        log_p = []
        gradients = []       

        for particle in value:
            
            nn.utils.vector_to_parameters(particle, self.net.parameters())
            log_p_prior = self.prior.log_prob(particle)

            ind = np.random.choice(len(self.y), size=self.b_size, replace=False)
            output = torch.sigmoid(self.net(self.X[ind]))
            loss = self.loss(output, self.y[ind].reshape(output.shape))
            
            grad = autograd.grad(loss+log_p_prior, self.net.parameters())
            w_grad_list = []
            for g in grad:
                w_grad_list.append(g.flatten())
            
            gradients.append(torch.cat(w_grad_list))
            log_p.append(loss+log_p_prior)
        
        grads = torch.stack(gradients)

        return torch.stack(log_p), grads



# Energy based Probability Distribution : Entropy of NN prediction
class Energy():
    def __init__(self,input_shape, net,prior,device=None, temp=20, use_prior=False, post_nn=None, ag_kde=None, beta=None, only_prior=False,env=None, goal_pred=False):
        self.input_shape = input_shape
        self.net = net
        self.temp = temp
        self.use_prior = use_prior
        self.prior_distrib = prior
        self.post_nn = post_nn 
        self.device = 'cpu' if device is None else device
        self.ag_kde = ag_kde
        self.lamb = 0.1
        self.beta = beta
        self.only_prior = only_prior
        self.fact_prob = 0
        self.bern = True
        self.annealed_energy = False
        self.env = env
        self.goal_pred=goal_pred


    def log_prob(self, value, log=True, use_prior=None):
        
        if self.use_prior:
            prior_log_prob = self.prior_distrib.log_prob(value)
            if len(prior_log_prob.shape) == 1:
                prior_log_prob = prior_log_prob.unsqueeze(1)

        if self.only_prior or self.annealed_energy:
            return prior_log_prob if log else torch.exp(prior_log_prob)



        if self.goal_pred:
            init_obs_batch = torch.from_numpy(self.env.reset()["achieved_goal"]).repeat(value.shape[0],1)
        else:
            init_obs_batch = torch.from_numpy(self.env.reset()["observation"]).repeat(value.shape[0],1)

        #import ipdb;ipdb.set_trace()
        #X = torch.cat((torch.zeros(value.shape).to(self.device),value),dim=1) # always start at point [0,0]
        X = torch.cat((init_obs_batch.to(self.device),value),dim=1)
        probas = torch.sigmoid(self.net(X))

        

        self.net_prob = probas

        if self.beta is not None:
            entropy = torch.exp(self.beta.log_prob(probas))

        elif not self.bern:
            c_1 = -probas*torch.log(probas + 1e-7)
            c_2 = -(1-probas)*torch.log(1-probas + 1e-7)

            if c_1.isnan().any() or c_2.isnan().any():
                import ipdb;ipdb.set_trace()
            entropy = c_1 + c_2
        else:
            d = torch.distributions.bernoulli.Bernoulli(probas)
            entropy = d.entropy()

        
        # Achieved Goal KDE log prob
        p_ag = 0 if self.ag_kde is None else self.ag_kde.score_samples_1(value).unsqueeze(1)

        if log:
            prob = self.temp*(entropy + self.fact_prob*torch.log(probas))
        else:
            prob = torch.exp(self.temp*(entropy + self.fact_prob*torch.log(probas)))

        #print(use_prior,self.beta,log)
        #traceback.print_stack()

        u_p = self.use_prior if use_prior is None else use_prior
        if not u_p:
            return prob
        else:
            return prob + prior_log_prob if log==True else prob*torch.exp(prior_log_prob)
        



class MultivariateGeneralizedGaussian():

    def __init__(self,mean=0,input_shape=1,alpha=1,beta=2, device=None):
        self.input_shape = input_shape
        self.mean = mean
        self.alpha = alpha
        self.beta = beta
        self.device = 'cpu' if device is None else device
    
    def log_prob(self, x, beta=None, alpha=None):

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
            cov = alpha*torch.eye(in_s).to(self.device)
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
            res_2 = torch.bmm(res_1,(x-mean).transpose(1,2))

            #prob = torch.exp(-1/2*res_2**beta)
            prob = -1/2*res_2**beta

            return (norm + prob).squeeze(1)
        


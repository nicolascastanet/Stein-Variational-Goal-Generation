import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from imblearn.over_sampling import RandomOverSampler




class LogisticRegression(torch.nn.Module):
    """ A Logistic Regression Model with sigmoid output in Pytorch"""
    def __init__(self, input_size):
        super().__init__()
        self.in_size = input_size
        self.w = torch.nn.Parameter(torch.randn((1, input_size)))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        if self.in_size > 1:
            s = torch.mm(self.w, x.T) + self.b 
        else:
            s = self.w*x + self.b
        return torch.sigmoid(s)

    def reset_weights(self):
        self.w = torch.nn.Parameter(torch.randn((1, self.in_size)))
        self.b = torch.nn.Parameter(torch.randn(1))


class OCSVM(torch.nn.Module):
    """
    PyTorch implementation on One class SVM with probability calibration
    """
    def __init__(self, kernel,sk_model, device):
        super(OCSVM, self).__init__()
        self.kernel = kernel
        self.sk_model = sk_model
        self.device = device
        self.logistic = LogisticRegression(input_size=1)
        self.logistic.to(device)
        self.fit_status = False

    def fit(self,X):
        """input : np.array"""

        # Fit One class SVM
        
        self.sk_model.fit(X)
        self.X_supp = torch.from_numpy(self.sk_model.support_vectors_).type(torch.float).to(self.device) # support vectors
        self.A = torch.from_numpy(self.sk_model.dual_coef_).type(torch.float).to(self.device) # alpha dual coef
        self.B = torch.from_numpy(self.sk_model._intercept_).type(torch.float).to(self.device) # decision biais
        self.n = int(self.sk_model.n_support_)

        # Fit logistic regression for calibration

        # prepare data with random oversample
        oversample = RandomOverSampler()
        X_train, y_train = self.sk_model.decision_function(X), self.sk_model.predict(X)
        X_train, y_train = oversample.fit_resample(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_train = (y_train+1)/2
        X_train, y_train = torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device)
        X_train, y_train = X_train.type(torch.float), y_train.type(torch.float)
        torch_train_dataset = data.TensorDataset(X_train,y_train)
        train_dataloader = data.DataLoader(torch_train_dataset, batch_size=len(torch_train_dataset))

        # Fit logistic regression model
        self.logistic.reset_weights()
        self.logistic.to(self.device)
        self.logistic.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.logistic.parameters(), lr=0.1)

        for _ in range(200):
            for x, y in train_dataloader:

                output = self.logistic(x)
                loss = criterion(output, y.reshape(-1,1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.fit_status = True

    def forward(self, x):
        """input : torch.tensor"""

        #import ipdb;ipdb.set_trace()

        #if self.sk_model.fit_status_:
        #    raise NameError('Sklean model not fitted !')
        
        K = self.kernel(x,self.X_supp) # kernel matrix
        scores = torch.mm(K,self.A.reshape(self.n,-1)) + self.B # decision function
        return torch.sign(scores), scores

    def log_prob(self,x,log=True):
        
        if log:
            prob = torch.log(self.logistic(self.forward(x)[1]))
        else:
            prob = self.logistic(self.forward(x)[1])
        return prob
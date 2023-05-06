from re import X
from h11 import Data
from matplotlib.pyplot import yscale
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import time


class RND(nn.Module):
    def __init__(self, input_size, output_size,device='cuda'):
        super(RND, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device


        self.mse = nn.MSELoss(reduce=False)
        self.sum_mse = nn.MSELoss()
   
        self.predictor = nn.Sequential(
            nn.Linear(input_size,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,output_size),
        )
        self.target = nn.Sequential(
            nn.Linear(input_size,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,output_size),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.xavier_uniform_(p.weight,gain=1)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.xavier_uniform_(p.weight, gain=1)
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

        self.average = 0.02
 


    def forward(self,obs):
        target_feature = self.target(obs)
        predict_feature = self.predictor(obs)
        diff = torch.sum(self.mse(predict_feature,target_feature),dim=-1)
        
        result = diff/self.average
        # if len(diff.shape)>1:
        #     print(diff.shape)
        # self.n += np.prod(diff.shape)
        # self.n_sum += diff.sum()
        # self.n_square_sum += torch.pow(diff,2).sum() 
       
        # running_mean = self.n_sum / self.n

        # running_std = torch.sqrt((self.n_square_sum-self.n_sum*self.n_sum/self.n)/(self.n-1))

        # result = 1+ (diff-running_mean)/(running_std+0.001)
        #print(result)
        return result

    def update(self,obs):
        # obs 20,64,3,2000
        #print(obs.shape)
        dataset = RNDDataset(obs.view(-1,self.input_size))
        dataloader = DataLoader(dataset,batch_size=128,shuffle=True)
        total_loss = 0
        for _ in range(10):
            for j,data in enumerate(dataloader):
                x  = data
                target  = self.target(x)
                predict = self.predictor(x)
                loss = self.sum_mse(target,predict)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss
            
        # target_feature = self.target(obs)
        # predict_feature = self.predictor(obs)
        # loss =  self.sum_mse(predict_feature,target_feature)

        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        # self.optimizer.step()
        return total_loss



class MetricDataset(Dataset):
    
    def __init__(self, x,y,label):

        super().__init__()

        self.x = x
        self.y = y
        self.label = label

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]

class RNDDataset(Dataset):
        
    def __init__(self, x):

        super().__init__()

        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]
    
    


from numpy import add
import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Predictor_Network(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_agents, lr=1e-5):
        super(Predictor_Network, self).__init__()

        input_dim = input_dim - n_agents *2 
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, n_agents)
        self.obs_size = input_dim
        self.nb_agent = n_agents

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.CE = nn.CrossEntropyLoss()
        self.CEP = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, softmax=True):
        x = x[...,self.nb_agent*2:]
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        if not softmax: return x 
        return  torch.softmax(x, dim=-1)

    def update(self, input):
        add_id = torch.eye(self.nb_agent).to(input.device).expand(  # 20 64 3
            [input.shape[0], input.shape[1], self.nb_agent, self.nb_agent])

        _obs = input.view(-1, self.obs_size+2*self.nb_agent)
        _label = add_id.reshape(-1, self.nb_agent).float()
     
        total_loss = 0
        for _ in range(1):
            for index in BatchSampler(SubsetRandomSampler(range(_obs.shape[0])), 256, False):
                logits = self.forward(_obs[index],softmax=False)
                softmax = F.softmax(logits, dim=1)
                log_softmax = F.log_softmax(logits, dim=1)
                
                entropy = -(log_softmax * softmax).sum(dim=-1)
                loss = torch.sum(self.CEP(softmax, _label[index]) + 0.05*entropy)
                #print(entropy.shape,loss.shape)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
                self.optimizer.step()
                total_loss += loss
        return loss

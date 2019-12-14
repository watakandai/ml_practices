##################################################
# @copyright Kandai Watanabe
# @email kandai.wata@gmail.com
# @institute University of Colorado Boulder
##################################################
import os 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tensors import var, get_tensor

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(TD3Actor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach(), requires_grad=False)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))

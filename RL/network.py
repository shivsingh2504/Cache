import torch
import torch.nn as nn
from RL.config import NetworkConfig


class QNetwork(nn.Module):
  def __init__(self,config:NetworkConfig)->None:
    
    super().__init__()
    self.config = config
    self.layers : nn.Module = nn.Identity()
    
  def forward(self,state:torch.Tensor)->torch.Tensor:
    raise NotImplementedError
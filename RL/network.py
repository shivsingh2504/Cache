import torch
import torch.nn as nn
from RL.config import NetworkConfig


class QNetwork(nn.Module):
  def __init__(self,config:NetworkConfig)->None:
    
    super().__init__()
    self.config = config
    input_dim = config.cache_capacity * config.num_state_features
    output_dim = config.cache_capacity
    layer_size = [input_dim,*config.hidden_sizes,output_dim]
    layers : list[nn.Module] = []
    for i in range(len(layer_size)-1):
      layers.append(nn.Linear(layer_size[i],layer_size[i+1]))
      is_last_layer = i ==len(layer_size)-2
      if not is_last_layer:
        layers.append(nn.ReLU())
    self.layers : nn.Module = nn.Sequential(*layers)
    
    
  def forward(self,state:torch.Tensor)->torch.Tensor:
    batch_size = state.shape[0]
    flattened = state.reshape(batch_size,-1)
    return self.layers(flattened)
import torch 
import torch.nn as nn

from lstm.config import ModelConfig

class PopularityPredictor(nn.Module):
  def __init__(self,config:ModelConfig)->None:
    super().__init__()
    self.config = config
    self.context_encoder = nn.LSTM(input_size=config.event_features,hidden_size=config.lstm_hidden_size,num_layers=config.lstm_layers,batch_first=True,dropout=config.dropout if config.lstm_layers > 1 else 0.0)
    
    self.candidate_encoder = nn.Sequential(
      nn.Linear(config.candidate_features,config.candidate_hidden_size),nn.ReLU()
    )
    
    fusion_input_size = config.lstm_hidden_size  + config.candidate_hidden_size
    
    self.fusion_head = nn.Sequential(
      nn.Linear(fusion_input_size,config.fusion_hidden_size),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(config.fusion_hidden_size,1)
    )
    
  def forward(self,context:torch.Tensor , candidate: torch.Tensor)->torch.Tensor:
    _, (h_n,_) = self.context_encoder(context)
    context_repr = h_n[-1]
    candidate_repr = self.candidate_encoder(candidate)
    fused = torch.cat([context_repr,candidate_repr],dim=1)
    return self.fusion_head(fused).squeeze(-1)
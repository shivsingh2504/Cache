import torch

from lstm.model import PopularityPredictor
from model_input.candidate_builder import CandidateFeatureBuilder
from feature.state import CacheState


class Predictor:
  def __init__(self,model:PopularityPredictor,state:CacheState,device:str="cpu")->None:
    self.model = model.to(device)
    self.model.eval()
    self.device = device
    self.candidate_builder = CandidateFeatureBuilder(state)
    @torch.no_grad()
    def score(self,context:torch.Tensor,candidate_keys:list[int])->dict[int,float]:
      if not candidate_keys:
        return {}
      candidate_features = [self.candidate_builder.build(key) for key in candidate_keys]
      candidate_tensor = torch.stack([torch.tensor([cf.frequency,cf.recency,cf.key_age],dtype=torch.float32)for cf in candidate_features]).to(self.device)
      context_batch = (context.unsqueeze(0).expand(len(candidate_keys),-1,1).to(self.device))
      predictions = self.model(context_batch,candidate_tensor)
      return{
        key: pred.item()
        for key,pred in zip(candidate_keys,predictions)
      }
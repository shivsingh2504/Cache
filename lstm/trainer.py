import torch
import torch.nn as nn
from torch.utils.data import DataLoader,DataChunk,Dataset,Subset

from lstm.config import TrainerConfig
from lstm.model import PopularityPredictor
from model_input.schema import TrainingSample

def inspect_label_distribution(samples:list[TrainingSample])->dict[str,float]:
  labels = torch.tensor([s.label for s in samples],dtype=torch.float32)
  return{
    "count": len(labels),
    "mean" : labels.mean().item(),
    "std" : labels.std().item(),
    "min" : labels.min().item(),
    "max" : labels.max().item(),
    "p50" : labels.quantile(0.50).item(),
    "p00" : labels.quantile(0.90).item(),
    "p99" : labels.quantile(0.99).item(),
    "frac_zero" : (labels==0).float().mean().item()
  } 
  
class TrainingSampleDataset(Dataset):
  def __init__(self,samples:list[TrainingSample])->None:
    self.samples = samples
    
  def __len__(self)->int:
    return len(self.samples)
  
  def __getitem__(self, ind:int)-> TrainingSample:
    return self.samples[ind]
  
def collate_fn(batch:list[TrainingSample])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
  contexts = torch.stack([torch.as_tensor(s.context,dtype=torch.float32) for s in batch])
  candidates = torch.stack([torch.as_tensor(s.candidate, dtype=torch.float32) for s in batch])
  labels = torch.tensor([s.label for s in batch], dtype=torch.float32)
  return contexts, candidates, labels

class Trainer:
  def __init__(self,model:PopularityPredictor,config:TrainerConfig)->None:
    self.model = model.to(config.device)
    self.config = config
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config.learning_rate)
    self.loss_fn = nn.MSELoss()
    
  def fit(self,samples: list[TrainingSample])->dict[str,list[float]]:
    dataset = TrainingSampleDataset(samples)
    val_size = int(len(dataset)* self.config.val_split)
    train_size = len(dataset) - val_size
    train_set,val_set = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_set,batch_size=self.config.batch_size,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_set,batch_size=self.config.batch_size,shuffle=False,collate_fn=collate_fn)
    history : dict[str,list[float]] = {"train_loss":[],"val_loss":[]}    
    for epoch in range(self.config.num_epochs):
      train_loss = self._run_epoch(train_loader,train=True)
      val_loss = self._run_epoch(val_loader,train=False)
      history["train_loss"].append(train_loss)
      history["train_loss"].append(train_loss)
      print(f"epoch {epoch + 1}/{self.config.num_epochs} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    return history
  

  def _run_epoch(self,loader:DataLoader,train:bool)->float:
    self.model.train(mode=train)
    total_loss = 0.0
    total_examples = 0
    
    with torch.set_grad_enabled(train):
      for contexts,candidates,labels in loader:
        contexts = contexts.to(self.config.device)
        candidates = candidates.to(self.config.device)
        labels = labels.to(self.config.device)
        
        predictions = self.model(contexts,candidates)
        loss = self.loss_fn(predictions,labels)
        
        if train:
          self.optimizer.zero_grad()
          loss.backward()
          if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(),self.config.grad_clip_norm)
          self.optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples
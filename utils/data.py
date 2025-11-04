import torch

def split(dataset, t=0.7, v=0.2, seed=42, to_torch = True, device = "cpu"):
  
  """
  splits the dataset and target lists in train, val and test. 
  if to_torch is True it also transform them into tensors
  """
  
  import random
  random.seed(42)
  n = len(dataset)
  n_train = int(n * t)
  n_val = int(n * v)
  
  indices = list(range(n))
  random.shuffle(indices)

  train_dataset = [dataset[i] for i in indices[:n_train]]
  val_dataset = [dataset[i] for i in indices[n_train:n_train + n_val]]
  test_dataset = [dataset[i] for i in indices[n_train + n_val:]]

  if to_torch:
    train_dataset = [torch.tensor(seq, dtype=torch.long).to(device) for seq in train_dataset]
    val_dataset = [torch.tensor(seq, dtype=torch.long).to(device) for seq in val_dataset]
    test_dataset = [torch.tensor(seq, dtype=torch.long).to(device) for seq in test_dataset]    
    
  return train_dataset, val_dataset, test_dataset


class SLM_dataset(torch.utils.data.Dataset):
  def __init__(self, dataset, context_size):
        
    super().__init__()
    target = []
    input_dataset = []
    masks = []
    
    for dialog in dataset:
        dialog = dialog.cpu()
      
        padding_mask = [1]* context_size

        pad_input = torch.tensor([0] * context_size)
        pad_target = torch.tensor([-100]*context_size)
      
        input_seq = pad_input.clone()
        target_seq = pad_target.clone()

        padding_mask[0:len(dialog)-1] = [0] * (len(dialog[0: context_size+1])-1)
        masks.append(torch.tensor(padding_mask, dtype=torch.bool))
      
        input_seq[0:len(dialog)] = dialog[0: context_size]
        input_dataset.append(input_seq)
        
        target_seq[0:len(dialog) -1] = dialog[1: context_size + 1]
        target.append(target_seq)

        
          
        for i in range(1, len(dialog) - context_size):
    
            input_seq = dialog[i:i + context_size]
            input_dataset.append(input_seq)
            
            target_seq = dialog[i + 1:i + context_size + 1]
            target.append(target_seq)
          

    self.dataset = input_dataset
    self.target = target
    self.masks = masks
    self.context_size = context_size
    
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return {"x":self.dataset[idx], "y":self.target[idx], "z": self.masks[idx]}

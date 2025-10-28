
def split(dataset, target, t=0.7, v=0.2, seed=42, to_torch = True, device = "cpu"):
  
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
  
  train_target = [target[i] for i in indices[:n_train]]
  val_target = [target[i] for i in indices[n_train:n_train + n_val]]
  test_target = [target[i] for i in indices[n_train + n_val:]]
  
  train_dataset = [dataset[i] for i in indices[:n_train]]
  val_dataset = [dataset[i] for i in indices[n_train:n_train + n_val]]
  test_dataset = [dataset[i] for i in indices[n_train + n_val:]]

  if to_torch:
    train_dataset = [torch.tensor(seq, dtype=torch.long).to(device) for seq in train_dataset]
    val_dataset = [torch.tensor(seq, dtype=torch.long).to(device) for seq in val_dataset]
    test_dataset = [torch.tensor(seq, dtype=torch.long).to(device) for seq in test_dataset]    
    
    train_target = [torch.tensor(seq, dtype=torch.long).to(device) for seq in train_target]
    val_target = [torch.tensor(seq, dtype=torch.long).to(device) for seq in val_target]
    test_target = [torch.tensor(seq, dtype=torch.long).to(device) for seq in test_target]
    
  return train_dataset, val_dataset, test_dataset, train_target, val_target, test_target


class SLM_dataset(torch.utils.data.Dataset):
  def __init__(self, dataset, target):
    super().__init__()
    self.dataset = dataset
    self.target = target
    
  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    return {"x":self.dataset[idx], "y":self.target[idx]}

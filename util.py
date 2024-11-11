import torch
import os
import torch.nn.functional as F
import wandb


def calculate_entropy(log_softmax_output):
    # Convert log-softmax output to softmax probabilities
    softmax_output = torch.exp(log_softmax_output)
    
    # Calculate entropy
    entropy = -torch.sum(softmax_output * log_softmax_output, dim=1)
    return entropy

def get_mask_indices(mask):
    return torch.where(mask)[0]

def test_time_entropy(model,data, augmentation_fn):
    data_tmp = augmentation_fn(data)
    out = model(data_tmp)
    entropy = calculate_entropy(out)
    return entropy

def test_time_entropy_latent_space(model, data,augmentation_fn):
    out = augmentation_fn(model,data)
    entropy = calculate_entropy(out)
    return entropy

def generate_random_data_split(dataset, train_init_size, val_size):
    generator = torch.Generator()
    generator.manual_seed(42)

    test_mask = dataset.test_mask
    test_mask.sum()

    val_pool = ~dataset.test_mask

    indices_pool =  torch.where(val_pool)[0]
    perm =  torch.randperm(indices_pool.shape[0], generator=generator)[:val_size]
    val_indices = indices_pool[perm]
    val_mask = torch.zeros(test_mask.shape[0], dtype=torch.bool)
    val_mask[val_indices] = True

    train_pool = ~dataset.test_mask & ~val_mask
    train_mask = torch.zeros(test_mask.shape[0], dtype=torch.bool)
    train_pool_idxs = torch.where(train_pool)[0]
    perm = torch.randperm(train_pool_idxs.shape[0], generator=generator)[:train_init_size]
    init_train_indices = train_pool_idxs[perm]
    train_mask[init_train_indices] = True
    train_pool[init_train_indices] = False
    
    return train_pool,train_mask, val_mask, test_mask



def generate_balanced_data_split(dataset, val_size, train_sample_per_class = 1, seed = 42):
    data = dataset[0]
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_classes = dataset.num_classes


    test_mask = dataset.test_mask
    test_mask.sum()

    val_pool = ~dataset.test_mask

    indices_pool =  torch.where(val_pool)[0]
    perm =  torch.randperm(indices_pool.shape[0], generator=generator)[:val_size]
    val_indices = indices_pool[perm]
    val_mask = torch.zeros(test_mask.shape[0], dtype=torch.bool)
    val_mask[val_indices] = True

    train_pool = ~dataset.test_mask & ~val_mask
    train_pool_indices = get_mask_indices(train_pool)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    y_pool = dataset.y[train_pool_indices]
    
    for i in range(num_classes):
        i_indices = torch.where(y_pool ==i)[0]
        perm = torch.randperm(i_indices.shape[0], generator=generator)[train_sample_per_class]
        train_mask[train_pool_indices[i_indices[perm]]] = True
        train_pool[train_pool_indices[i_indices[perm]]] = False

    return train_pool,train_mask, val_mask, test_mask

def generate_balanced_data_splits(dataset, num_splits, path):
    for i in range(num_splits):
        train_pool,train_mask, val_mask, test_mask = generate_balanced_data_split(dataset, 500, 1, seed = i) 
        data_clone = dataset[0].clone()
        data_clone.train_mask = train_mask
        data_clone.val_mask = val_mask
        data_clone.test_mask = test_mask
        data_clone.train_pool = train_pool
        
        full_path = os.path.join(path, f"split_{i}.pt")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(data_clone, full_path)
        

def nt_xent_loss(x1,x2,t=1.0):
    D, s_12, s_11, s_22 = sim_vectors(x1,x2,t)
    
    l1 = D / (D + s_12 + s_11)
    loss1 = -torch.log(l1)
    
    l2 = D / (D + s_12 + s_22)
    loss2 = -torch.log(l2)
    
    loss = (loss1 + loss2) / (2*len(x1))
    loss = loss.sum()
    return loss

def sim_vectors(x1,x2, t=1.0):
    sims_12 = tempered_cosine_similarity(x1,x2,t)
    sims_11 = tempered_cosine_similarity(x1,x1,t)
    sims_22 = tempered_cosine_similarity(x2,x2,t)
    
    D = torch.diag(sims_12,0)
    s_12 = sims_12.sum(dim=1) - D
    s_11 = sims_11.sum(dim=1) - torch.diag(sims_11,0)
    s_22 = sims_22.sum(dim=1) - torch.diag(sims_22,0)
    
    return D, s_12, s_11, s_22


    

def tempered_cosine_similarity(x1,x2, t=1.0):
    cos_sim = F.cosine_similarity(x1[None,:,:], x2[:,None,:], dim=-1)
    cos_sim = torch.exp(cos_sim / t)
    return cos_sim

class InfoNCE():
    def __init__(self, tau):
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob
        loss = loss.sum(dim=1)/anchor.size(0)
        return -loss.mean()
    
    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss
    

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()



def init_wandb(query_strategy,run):
  config={
    "learning_rate": 0.01,
    "architecture": "GCN",
    "dataset": "CORA",
    "epochs": 200,
    "strategy": str(query_strategy)
    }
  
  if hasattr(query_strategy, "augmentation_fn"):
    augmentation = query_strategy.augmentation_fn
    config["augmentations"] = []
    if isinstance(augmentation, T.Compose):
      transforms = augmentation.transforms
    else:
      transforms = [augmentation]
      
    for t in transforms:
      aug_config = {}
      aug_config["name"] = str(t)
      aug_config["hyperparameters"] = t.__hyperparameters__()
      config["augmentations"].append(aug_config)
  
  wandb.init(
    # Set the project where this run will be logged
    project="graph-active-learning",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"{query_strategy}_{run}",
    # Track hyperparameters and run metadata
    config=config)
  return config
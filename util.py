import torch
from torch_geometric.datasets import Planetoid
import os

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
    train_pool = torch.ones(data.num_nodes, dtype=torch.bool)
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
        

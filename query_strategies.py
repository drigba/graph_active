from typing import Any
from util import *
import random
from scipy.stats import mode
import torch_geometric.transforms as T 


class QueryStrategy:
    def __call__(self,model,data, train_pool) -> Any:
        raise NotImplementedError
    
class AugmentedQueryStrategy(QueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10):
        super().__init__()
        self.augmentation_fn = augmentation_fn
        self.num_passes = num_passes
        
    def __str__(self) -> str:
        t = self.__concat_params__(str)
        s = f"{self.__class__.__name__}({t})"
        return s
    
    def __repr__(self) -> str:
        t = self.__concat_params__(repr)
        s = f"{self.__class__.__name__}(transforms={t})"
        return s
    
    def __concat_params__(self,fn):
        if isinstance(self.augmentation_fn, T.Compose):
          transforms = self.augmentation_fn.transforms
        else:
         transforms = [self.augmentation_fn]
        
        t = ", ".join([fn(transform) for transform in transforms])
        t = f"[{t}]"
        return t
    
class RandomQuery(QueryStrategy):
    def __init__(self):
        super().__init__()
        
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        chosen_ix = random.choice(pool_indices)
        return chosen_ix
    
    def __str__(self) -> str:
        return self.__class__.__name__
        
        
class EntropyQuery(QueryStrategy):
    def __init__(self):
        super().__init__()
        
    def __call__(self,model,data, train_pool):
        return self.entropy_query(model,data, train_pool)
    
    def entropy_query(self,model,data, train_pool):
        model.eval()
        out = model(data)
        entropy = calculate_entropy(out)
        pool_indices = get_mask_indices(train_pool)
        chosen_node_ix = torch.argmax(entropy[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
class AugmentGraphModeQuery(AugmentedQueryStrategy):

    def __call__(self, model, data, train_pool) -> Any:
        model.eval()
        chosen_nodes = []
        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            chosen_node = self.entropy_query(model,data_tmp, train_pool)
            chosen_nodes.append(chosen_node.item())
        chosen_node = mode(chosen_nodes)[0]
        return chosen_node

        
class AugmentGraphSumEntropyQuery(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight

        
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        entropy_sum = torch.zeros(data.num_nodes).to(data.x.device)

        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            out = model(data_tmp)
            entropy = calculate_entropy(out)
            entropy_sum += entropy
        entropy_sum /= self.num_passes
        orig_out = model(data)
        entropy = calculate_entropy(orig_out)
        entropy_sum += self.original_weight * entropy
            
        chosen_node_ix = torch.argmax(entropy_sum[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        
        return chosen_node
    
    def  __concat_params__(self,fn) -> str:
        return super().__concat_params__(fn) + f", original_weight={self.original_weight}"

    

    
    
class AugmentGraphLogitChange(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight
        
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        out = torch.zeros_like(model(data))
        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            out += model(data_tmp)
        out = out / self.num_passes
        
        orig_out = model(data)
        logit_change = out + self.original_weight * orig_out
        
        entropy = calculate_entropy(logit_change)
        chosen_node_ix = torch.argmax(entropy[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def  __concat_params__(self,fn) -> str:
        return super().__concat_params__(fn) + f", original_weight={self.original_weight}"
    


class AugmentGraphSumQueryLatent(AugmentedQueryStrategy):


    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        entropy_sum = torch.zeros(data.num_nodes).to(data.x.device)

        for _ in range(self.num_passes):
            latent = model.get_latent(data)
            latent_augmented = self.augmentation_fn(latent)
            out = model.predict_from_latent(latent_augmented)
            entropy = calculate_entropy(out)
            entropy_sum += entropy
            
        chosen_node_ix = torch.argmax(entropy_sum[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
        

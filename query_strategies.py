from typing import Any
from util import *
import random
from scipy.stats import mode
import torch_geometric.transforms as T 
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import to_dense_adj

class QueryStrategy:
    def __call__(self,model,data, train_pool) -> Any:
        raise NotImplementedError
    
    def __str__(self) -> str:
        return self.__class__.__name__
    def __repr__(self) -> str:
        return self.__class__.__name__
    
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
        out = model.test_step(data)
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
            out = model.test_step(data_tmp)
            entropy = calculate_entropy(out)
            entropy_sum += entropy
        entropy_sum /= self.num_passes
        orig_out = model.test_step(data)
        entropy = calculate_entropy(orig_out)
        entropy_sum += self.original_weight * entropy
            
        chosen_node_ix = torch.argmax(entropy_sum[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        
        return chosen_node
    
    def  __concat_params__(self,fn) -> str:
        return super().__concat_params__(fn) + f", original_weight={self.original_weight}"
    
class AugmentGraphSumEntropyQueryNeighborhood(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10,n_hop = 2, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight
        self.n_hop = n_hop
        
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        entropy_sum = torch.zeros(data.num_nodes).to(data.x.device)

        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            out = model.test_step(data_tmp)
            entropy = calculate_entropy(out)
            entropy_sum += entropy
        entropy_sum /= self.num_passes
        orig_out = model.test_step(data)
        entropy = calculate_entropy(orig_out)
        entropy_sum += self.original_weight * entropy
            
        
        adjaceny = to_dense_adj(data.edge_index)[0]
        adjaceny = adjaceny / adjaceny.sum(dim=1, keepdim=True)
        adjaceny = adjaceny + torch.eye(adjaceny.shape[0]).to(adjaceny.device)

        for _ in range(self.n_hop):
            entropy_sum = torch.matmul(adjaceny,entropy_sum)
        chosen_node_ix = torch.argmax(entropy_sum[pool_indices])
        
        chosen_node = pool_indices[chosen_node_ix]
        
        return chosen_node
    
    def  __concat_params__(self,fn) -> str:
        return super().__concat_params__(fn) + f", original_weight={self.original_weight}, n_hop={self.n_hop}"
    
    
class AugmentGraphRatioEntropyQuery(AugmentedQueryStrategy):

    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        entropy_sum = torch.zeros(data.num_nodes).to(data.x.device)

        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            out = model.test_step(data_tmp)
            entropy = calculate_entropy(out)
            entropy_sum += entropy
        entropy_sum /= self.num_passes
        
        orig_out = model.test_step(data)
        entropy = calculate_entropy(orig_out)
        entropy_ratio = entropy / (entropy_sum + 1e-6)
            
        chosen_node_ix = torch.argmax(entropy_ratio[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        
        return chosen_node
    
    


class ContrastiveMinMax(QueryStrategy):

    def __call__(self,model,data, train_pool):
        
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        out = model(data)
        
        dist = torch.cdist(out,out)
        dense_adj = to_dense_adj(data.edge_index)
        crit_matrix = dist * dense_adj
        
        max_crit = crit_matrix.max(dim=1).values[0]
            
        chosen_node_ix = torch.argmin(max_crit[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}"
        return s
    
    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}"
        return s
    
class ContrastiveMinAvg(QueryStrategy):

    def __call__(self,model,data, train_pool):
        
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        out = model(data)
        
        dist = torch.cdist(out,out)
        dense_adj = to_dense_adj(data.edge_index)
        crit_matrix = dist * dense_adj
        
        avg_crit = crit_matrix.mean(dim=1)[0]
            
        chosen_node_ix = torch.argmin(avg_crit[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}"
        return s
    
    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}"
        return s
    
class ContrastiveCentral(QueryStrategy):

    def __call__(self,model,data, train_pool):
        
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        out = model(data)
        
        dist = torch.cdist(out,out)

        
        sum_crit = dist.sum(dim=1)
            
        chosen_node_ix = torch.argmin(sum_crit[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}"
        return s
    
    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}"
        return s
    
    

    
    
class AugmentGraphLogitChange(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight
        
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        orig_out = model.test_step(data)
        out = torch.zeros_like(orig_out)
        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            out += model.test_step(data_tmp)
        out = out / self.num_passes
        
        
        logit_change = out + self.original_weight * orig_out
        
        entropy = calculate_entropy(logit_change)
        chosen_node_ix = torch.argmax(entropy[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def  __concat_params__(self,fn) -> str:
        return super().__concat_params__(fn) + f", original_weight={self.original_weight}"
    


class AugmentGraphSumQueryLatent(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight

    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        entropy_sum = torch.zeros(data.num_nodes).to(data.x.device)
        latent = model(data)
        
        orig_out = model.predict_log_proba(latent.detach().cpu().numpy())
        orig_out = torch.tensor(orig_out).to(data.y.device)
        entropy = calculate_entropy(orig_out)

        for _ in range(self.num_passes):
            latent_tmp = latent.clone()
            latent_augmented = self.augmentation_fn(latent_tmp)
            latent_augmented = latent_augmented.detach().cpu().numpy()
            
            out = model.predict_log_proba(latent_augmented)
            out = torch.tensor(out).to(data.y.device)
            entropy = calculate_entropy(out)
            entropy_sum += entropy
        
        entropy_sum /= self.num_passes
        entropy_sum += self.original_weight * entropy
            
        chosen_node_ix = torch.argmax(entropy_sum[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    
class AugmentGraphExpectedGraph(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight
        
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool).cpu()
        model.eval()
        orig_latent = model(data).detach().cpu()
        latent = torch.zeros_like(orig_latent).cpu()
        for _ in range(self.num_passes):
            data_tmp = data.clone()
            data_tmp = self.augmentation_fn(data_tmp)
            latent += model(data_tmp).detach().cpu()
            
        latent = latent / self.num_passes
        expected_latent = latent + self.original_weight * orig_latent
        out = model.predict_log_proba(expected_latent.detach().cpu().numpy())
        entropy = calculate_entropy(torch.tensor(out))
        
        chosen_node_ix = torch.argmax(entropy[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node
    
    def  __concat_params__(self,fn) -> str:
        return super().__concat_params__(fn) + f", original_weight={self.original_weight}"
    
    
    
class DropNodeLatentDistance(QueryStrategy):

    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        latent = model(data)
        min_dist = 0
        min_ix = pool_indices[0]
        
        for i in pool_indices:
            data_tmp = data.clone()
            mask = (data_tmp.edge_index[0] != i) & (data_tmp.edge_index[1] != i)
            data_tmp.edge_index = data_tmp.edge_index[:, mask]
            
            out = model(data_tmp)
            dist = -torch.sqrt(torch.pow(latent-out,2.0).sum())
            if dist < min_dist:
                min_dist = dist
                min_ix = i
                
        return min_ix
    
class DropNodeLossChange(QueryStrategy):

    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        latent = model(data)
        min_loss = 0
        min_ix = pool_indices[0]
        
        
        for i in pool_indices:
            data_tmp = data.clone()
            mask = (data_tmp.edge_index[0] != i) & (data_tmp.edge_index[1] != i)
            data_tmp.edge_index = data_tmp.edge_index[:, mask]
            latent2 = model(data_tmp)
            loss = -model.model.loss(latent,latent2)
            
            if loss < min_loss:
                min_ix = i
                min_loss = loss
                
        return min_ix
                
class DropNodeAccChange(QueryStrategy):

    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        min_acc = 0
        min_ix = pool_indices[0]
        
        
        for i in pool_indices:
            data_tmp = data.clone()
            mask = (data_tmp.edge_index[0] != i) & (data_tmp.edge_index[1] != i)
            data_tmp.edge_index = data_tmp.edge_index[:, mask]
            latent2 = model(data_tmp)
            latent2 =latent2[data.train_mask].detach().cpu().numpy()
            labels =data.y[data.train_mask].detach().cpu().numpy()
            acc = - model.predictor.score(latent2,labels)
            
            if acc < min_acc:
                min_ix = i
                min_acc = acc
                
        return min_ix
        

       
        
        
        

class AugmentGraphLogitChangeLatent(AugmentedQueryStrategy):
    def __init__(self,augmentation_fn, num_passes=10, original_weight = 0.0):
        super().__init__(augmentation_fn, num_passes)
        self.original_weight = original_weight
    def __call__(self,model,data, train_pool):
        pool_indices = get_mask_indices(train_pool)
        model.eval()
        latent = model(data)
        orig_out = model.predict_log_proba(latent.detach().cpu().numpy())
        orig_out = torch.tensor(orig_out).to(data.y.device)
        
        outs = torch.zeros_like(orig_out)
        for _ in range(self.num_passes):
            latent_tmp = latent.clone()
            latent_augmented = self.augmentation_fn(latent_tmp)
            latent_augmented = latent_augmented.detach().cpu().numpy()
            out = model.predict_log_proba(latent_augmented)
            out = torch.tensor(out).to(data.y.device)
            outs += out
        outs = outs / self.num_passes
        logit_change = outs + self.original_weight * orig_out
        
        entropy = calculate_entropy(logit_change)
            
        chosen_node_ix = torch.argmax(entropy[pool_indices])
        chosen_node = pool_indices[chosen_node_ix]
        return chosen_node

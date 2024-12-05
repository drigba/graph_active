from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature
import torch
from torch_geometric.transforms import BaseTransform
from util import drop_edge_weighted, drop_feature_weighted_2

   

class DropEdge(BaseTransform):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        data.edge_index, _ = dropout_edge(data.edge_index, p=self.p)
        return data
    
    def __hyperparameters__(self):
        return {'p': self.p}
    
    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)

    def __str__(self) -> str:
        return self.__class__.__name__
    
    
class DropEdgeWeighted(BaseTransform):
    def __init__(self, weights, p, threshold=1.0):
        self.p = p
        self.weights = weights
        self.threshold = threshold

    def __call__(self, data):
        data.edge_index = drop_edge_weighted(data.edge_index, self.weights, p=self.p, threshold=self.threshold)
        return data
    
    def __hyperparameters__(self):
        return {'p': self.p,
                'th': self.threshold}
    
    def __repr__(self):
        return '{}(p={}, th={})'.format(self.__class__.__name__, self.p, self.threshold)

    def __str__(self) -> str:
        return self.__class__.__name__
    
class MaskFeature(BaseTransform):
    def __init__(self, p, mode='col'):
        self.p = p
        if mode not in ['col', 'all', 'row']:
            raise ValueError('Invalid mode. Choose from col, all, row')
        self.mode = mode

    def __call__(self, data):
        data.x, _ = mask_feature(data.x, p=self.p, mode=self.mode)
        return data
    
    def __hyperparameters__(self):
        return {'p': self.p,
                'mode': self.mode}
        
    def __repr__(self):
        return '{}(p={}, mode={})'.format(self.__class__.__name__, self.p, self.mode)

    def __str__(self) -> str:
        return self.__class__.__name__
    
    
class MaskFeatureWeighted(BaseTransform):
    def __init__(self, weights,p,  threshold=0.7):
        self.p = p
        self.weights = weights
        self.threshold = threshold


    def __call__(self, data):
        data.x = drop_feature_weighted_2(data.x, self.weights, p=self.p, threshold=self.threshold)
        return data
    
    def __hyperparameters__(self):
        return {'p': self.p,
                'th': self.threshold}
        
    def __repr__(self):
        return '{}(p={}, th={})'.format(self.__class__.__name__, self.p, self.threshold)

    def __str__(self) -> str:
        return self.__class__.__name__
    
class NoiseFeature(BaseTransform):
    def __init__(self, noise_level, noise_prob = 1.0,mode = 'col'):
        if mode not in ['col',  'row']:
            raise ValueError('Invalid mode. Choose from col, row')
        self.noise_level = noise_level
        self.noise_prob = noise_prob
        self.mode = mode
    
    def __call__(self, data):
        if self.mode == 'col':
            noised_nodes_mask = torch.rand(data.x.shape[1]) < self.noise_prob
            data.x[:,noised_nodes_mask] = data.x[:,noised_nodes_mask] + torch.randn_like(data.x[:,noised_nodes_mask]) * self.noise_level
        elif self.mode == 'row':
            noised_nodes_mask = torch.rand(data.x.shape[0]) < self.noise_prob
            data.x[noised_nodes_mask] = data.x[noised_nodes_mask] + torch.randn_like(data.x[noised_nodes_mask]) * self.noise_level
        return data
    
    def __hyperparameters__(self):
        return {'noise_level': self.noise_level,
                'noise_prob': self.noise_prob,
                'mode': self.mode}
    
    def __repr__(self):
        return '{}(noise_level={}, noise_prob={}, mode={})'.format(self.__class__.__name__, self.noise_level,self.noise_prob, self.mode)

    def __str__(self) -> str:
        return self.__class__.__name__
    
class NoiseFeatureWeighted(BaseTransform):
    def __init__(self,weights, noise_level, th):
        self.noise_level = noise_level
        self.weights = weights
        self.th = th
        
        w = self.weights
        w = w / w.mean() * self.noise_level
        w = w.where(w < self.th, torch.ones_like(w) * self.th)
        self.weights = w
    
    def __call__(self, data):
        
        noise = torch.randn_like(data.x) * self.weights
        data.x = data.x + noise
        return data
    
    def __hyperparameters__(self):
        return {'noise_level': self.noise_level,
                'th': self.th}
    
    def __repr__(self):
        return '{}(noise_level={}, th={})'.format(self.__class__.__name__, self.noise_level,self.th)

    def __str__(self) -> str:
        return self.__class__.__name__
    


class NoiseLatent(BaseTransform):
    def __init__(self, noise_level):
        self.noise_level = noise_level
    
    def __call__(self, out):
        out = out + torch.randn_like(out) * self.noise_level
        return out
    
    def __hyperparameters__(self):
        return {'noise_level': self.noise_level}
    
    def __repr__(self):
        return '{}(noise_level={})'.format(self.__class__.__name__, self.noise_level)

    def __str__(self) -> str:
        return self.__class__.__name__
    

class COSTA(BaseTransform):
    def __call__(self, z):
        k = torch.tensor(int(z.shape[0]))
        p = (1/torch.sqrt(k))*torch.randn(k, z.shape[0]).to(z.device)
        
        out = p @ z
        return out        
    
    def __hyperparameters__(self):
        return {}
    




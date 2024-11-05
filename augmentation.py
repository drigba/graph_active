from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature
import torch
from torch_geometric.transforms import BaseTransform


   

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
    
class NoiseFeature(BaseTransform):
    def __init__(self, noise_level, noise_prob = 1.0,mode = 'col'):
        if mode not in ['col',  'row']:
            raise ValueError('Invalid mode. Choose from col, all, row')
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
    



# class AddNoiseAll(Augmentation):
#     def __init__(self, noise_level):
#         self.noise_level = noise_level
        

#     def _augment(self, data):
#         data.x = data.x + torch.randn_like(data.x) * self.noise_level
#         return data
        
# class AddNoiseCol(Augmentation):
#     def __init__(self, noise_level, noise_prob):
#         self.noise_level = noise_level
#         self.noise_prob = noise_prob
        
#     def _augment(self, data):
#         noised_nodes_mask = torch.rand(data.x.shape[1]) < self.noise_prob
#         data.x[:,noised_nodes_mask] = data.x[:,noised_nodes_mask] + torch.randn_like(data.x[:,noised_nodes_mask]) * self.noise_level
#         return data
    
# class AddNoiseNode(Augmentation):
#     def __init__(self, noise_level, noise_prob):
#         self.noise_level = noise_level
#         self.noise_prob = noise_prob
    
#     def _augment(self, data):
#         noised_nodes_mask = torch.rand(data.x.shape[0]) < self.noise_prob
#         data.x[noised_nodes_mask] = data.x[noised_nodes_mask] + torch.randn_like(data.x[noised_nodes_mask]) * self.noise_level
#         return data
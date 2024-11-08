import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F



class BaseGCN(torch.nn.Module):
    def __init__(self, convs, projection_head= torch.nn.Identity()):
        super(BaseGCN, self).__init__()
        
        if isinstance(convs, torch.nn.ModuleList):
            self.convs = convs
        elif isinstance(convs, list):
            self.convs = torch.nn.ModuleList()
            for conv in convs:
                self.convs.append(conv)
        elif isinstance(convs, torch.nn.Module):
            self.convs = torch.nn.ModuleList([convs])
            
        self.projection_head = projection_head
        
    def get_latent(self,data):
        x, edge_index = data.x, data.edge_index
        
        x = self.convs[0](x, edge_index)
        if len(self.convs) > 1:
            for conv in self.convs[1:]:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = conv(x, edge_index)
        
        return x
    
    def project(self,x):
        x = self.projection_head(x)
        return x
        
    def forward(self, data):
        x = self.get_latent(data)
        return self.project(x)
      
        
class GCNWithProjection(BaseGCN):
    def __init__(self, num_features,num_classes):
        conv1 = GCNConv(num_features, 16)
        conv2 = GCNConv(16, 16)
        linear = torch.nn.Linear(16, num_classes)
        super(GCNWithProjection, self).__init__([conv1,conv2], linear) 
    
class GCNContrastive(BaseGCN):
    def __init__(self, num_features):
        conv1 = GCNConv(num_features, 128)
        conv2 = GCNConv(128, 64)
        projection = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16)
        )
        super(GCNContrastive, self).__init__([conv1,conv2], projection)
    
class GCNContrastiveMini(BaseGCN):
    def __init__(self, num_features):
        conv1 = GCNConv(num_features, 64)
        conv2 = GCNConv(64, 2)
        super(GCNContrastiveMini, self).__init__([conv1,conv2])

class GCN(BaseGCN):
    def __init__(self, num_features,num_classes):
        conv1 = GCNConv(num_features, 16)
        conv2 = GCNConv(16, num_classes)
        super(GCN, self).__init__([conv1,conv2])

    

        
    
    






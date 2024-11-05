import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# COMBINE????
class GCN_with_linear(torch.nn.Module):
    def __init__(self, num_features,num_classes):
        super(GCN_with_linear, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.linear = torch.nn.Linear(16, num_classes)
        
    def get_latent(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def predict_from_latent(self,x):
        x = F.relu(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1) 

    def forward(self, data):
        x = self.get_latent(data)
        return self.predict_from_latent(x)
    
class GCN_Contrastive(torch.nn.Module):
    def __init__(self, num_features,num_classes):
        super(GCN_Contrastive, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.projection(x)
        return x
    
    

class GCN(torch.nn.Module):
    def __init__(self, num_features,num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

        
    
    






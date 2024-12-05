import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from typing import Callable, List, Optional



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

    
class GRACEEncoder(torch.nn.Module):
    """Implementation of the Encoder for GRACE"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Callable = F.relu,
        base_model=GCNConv,
        k: int = 2,
    ) -> None:
        """
        Initialize the encoder module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (Callable, optional): Activation function. Defaults to F.relu.
            base_model (torch.nn.Module, optional): Base model. Defaults to GCNConv.
            k (int, optional): Number of layers. Defaults to 2.
        """
        super(GRACEEncoder, self).__init__()
        self.base_model = base_model

        assert k >= 2, "k needs to be atleast 2"
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = torch.nn.ModuleList(self.conv)  # type: ignore[assignment]
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass through the encoder module.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Representations from the encoder module.
        """
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

        
    
    






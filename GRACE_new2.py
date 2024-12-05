from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge

from hivegraph.augmentations import drop_feature
from augmentation import *
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from util import *  


__all__: List[str] = ["GRACENew2"]


class GRACENew2(torch.nn.Module):
    """
    Implementation of deep GRAph Contrastive rEpresentation learning (GRACE).

    .. image:: ../../media/contrastive/grace.png

    References:

    * https://arxiv.org/abs/2006.04131v2
    * https://github.com/CRIPAC-DIG/GRACE
    * https://wandb.ai/sauravmaheshkar/GRACE/reports/Graph-Contrastive-Learning--Vmlldzo2MzYxODEy
    """

    def __init__(
        self,
        encoder_module: torch.nn.Module,
        projection_head: torch.nn.Module,
        augmentor1: BaseTransform,
        augmentor2: BaseTransform,
        tau: Optional[float] = 0.5,
        lambda_: Optional[float] = 1.0,
        model_name: str = "GRACENew2",
        use_labels: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the GRACE model.

        Args:
            num_features (int): Number of input features.
            hidden (int): Number of hidden units.
            num_layers (int): Number of layers.
            activation (Callable, optional): Activation function. Defaults to F.relu.
            base_model (torch.nn.Module, optional): Base model. Defaults to GCNConv.
            projection_dim (int, optional): Dimension of the projected representations.
                Defaults to 128.
            tau (Optional[float], optional): Temperature parameter. Defaults to 0.5.
            model_name (str, optional): Name of the model. Defaults to "GRACE".
        """
        super(GRACENew2, self).__init__(**kwargs)
        self.encoder_module = encoder_module
        self.tau = tau
        self.projection_head = projection_head

        self.augmentor1 = augmentor1
        self.augmentor2 = augmentor2
        
        self.use_labels = use_labels

    def train_step(
        self,
        data: Data
        ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            float: Loss.
        """
        # Generate Graph Views
        if self.use_labels:
            labels = label_indices(data)
        else:
            labels = None
        ## Removing Edges (RE)
        data_view1 = data.clone()
        data_view2 = data.clone()
        data_view1 = self.augmentor1(data_view1)
        data_view2 = self.augmentor2(data_view2)
        
        x_1, edge_index_1 = data_view1.x, data_view1.edge_index
        x_2, edge_index_2 = data_view2.x, data_view2.edge_index

        ## Generating views
        z1 = self.forward(x_1, edge_index_1)
        z2 = self.forward(x_2, edge_index_2)
        

        h1 = z1
        h2 = z2
        

        # Calculate Loss
        loss = self.loss(h1, h2, labels = labels, batch_size=0)

        return loss

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,labels = None) -> torch.Tensor:
        """
        Compute the "semi_loss" between two given views.

        Space Complexity: :math:`O(N^2)`

        .. math::

            l(u_i, v_i) = \log \\frac{e^{\\theta (u_i, v_i) / \\tau}}{e^{\\theta (u_i, v_i) / \\tau} + \color{blue}{\sum_{k=1}^{N} \mathbb{1}_{[k \\neq 1]} e^{\\theta(u_i, v_k)/ \\tau}} + \color{green}{\sum_{k=1}^{N}\mathbb{1}_{[k \\neq 1]} e^{\\theta(u_i, u_k)/ \\tau}} }

        * The equation in blue represents the loss between the inter-view negative pairs.
        * The equation in green represents the loss between the intra-view negative pairs.

        Args:
            z1 (torch.Tensor): First set of views.
            z2 (torch.Tensor): Second set of views.

        Returns:
            torch.Tensor: "semi_loss" between the two sets of views.
        """  # noqa: E501
        intraview_pairs = self.normalize_with_temp(self.compute_cosine_sim(z1, z1))
        interview_pairs = self.normalize_with_temp(self.compute_cosine_sim(z1, z2))
    
        if self.use_labels == False:
            return -torch.log(
                interview_pairs.diag()
                / (intraview_pairs.sum(1) + interview_pairs.sum(1) - intraview_pairs.diag())
            )
        else:
            loss_vector = torch.zeros_like(interview_pairs[0])
            for k,v in labels.items():
                s = interview_pairs[v][:,v].sum()
                loss_vector[v] = s / (len(v)+1)
            return -torch.log(
                (interview_pairs.diag() + self.lambda_ * loss_vector) /
                (intraview_pairs.sum(1)+ self.lambda_ * loss_vector + interview_pairs.sum(1) -  intraview_pairs.diag())
            )

    def batched_semi_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Calculate the "semi_loss" between a batch of views

        Space Complexity: :math:`O(BN)`

        Args:
            z1 (torch.Tensor): First batch of views.
            z2 (torch.Tensor): Second batch of views.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: "semi_loss" between the two batches.
        """
        # Helper variables
        device: torch.device = z1.device
        num_nodes: int = z1.size(0)
        num_batches: int = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        losses: List[torch.Tensor] = []

        for i in range(num_batches):
            # Mask out other values not in the current batch
            mask: torch.Tensor = indices[i * batch_size : (i + 1) * batch_size]

            # Similar to self.semi_loss()
            intraview_pairs: torch.Tensor = self.normalize_with_temp(
                self.compute_cosine_sim(z1[mask], z1)
            )  # (batch_size, num_nodes)
            interview_pairs: torch.Tensor = self.normalize_with_temp(
                self.compute_cosine_sim(z1[mask], z2)
            )  # (batch_size, num_nodes)

            current_batch_loss: torch.Tensor = -torch.log(
                interview_pairs[:, i * batch_size : (i + 1) * batch_size].diag()
                / (
                    intraview_pairs.sum(1)
                    + interview_pairs.sum(1)
                    - intraview_pairs[:, i * batch_size : (i + 1) * batch_size].diag()
                )
            )

            losses.append(current_batch_loss)

        assert (
            len(losses) == num_batches
        ), "Number of losses must equal number of batches"

        return torch.cat(losses)

    def loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mean: Optional[bool] = True,
        labels = None,
        batch_size: int = 0,
    ) -> torch.Tensor:
        r"""
        Compute the overall loss for all positive pairs.

        Eqn(2) from the paper.

        .. math::

            \mathcal{J} = \frac{1}{2N} \displaystyle \sum_{i=1}^{N} \left ( l(u_i, v_i) + l(v_i, u_i) \right)

        References:

        * https://arxiv.org/abs/2006.04131v2

        Args:
            z1 (torch.Tensor): First set of views.
            z2 (torch.Tensor): Second set of views.
            mean (bool, optional): If True, return the mean loss. Defaults to True.
            batch_size (int, optional): Batch size. Defaults to 0.

        Returns:
            torch.Tensor: Overall loss.
        """  # noqa: E501
        # Generate views
        # one is used as the anchor
        # other forms the positive sample
        u: torch.Tensor = self.project(z1)
        v: torch.Tensor = self.project(z2)

        # As the two views are symmetric
        # the other loss is just calculated
        # using alternate parameters

        if batch_size == 0:
            l1: torch.Tensor = self.semi_loss(u, v,labels)
            l2: torch.Tensor = self.semi_loss(v, u,labels)
        else:
            l1: torch.Tensor = self.batched_semi_loss(u, v, batch_size)  # type: ignore[no-redef]
            l2: torch.Tensor = self.batched_semi_loss(v, u, batch_size)  # type: ignore[no-redef]

        loss: torch.Tensor = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()

        return loss

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass through the encoder module.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Representations from the encoder module.
        """
        return self.encoder_module(x, edge_index)

    def compute_cosine_sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity between two sets of views.

        Args:
            z1 (torch.Tensor): First set of views.
            z2 (torch.Tensor): Second set of views.

        Returns:
            torch.Tensor: Cosine similarity between the two sets of views.
        """
        z1 = torch.nn.functional.normalize(z1)
        z2 = torch.nn.functional.normalize(z2)
        return torch.mm(z1, z2.T)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project the representations to a lower-dimensional space.

        This has been shown to enchance the expression power of the critic,
        For details refer to the section 3.2.1

        References:

        * https://arxiv.org/abs/2006.04131v2

        Args:
            z (torch.Tensor): Representations from the encoder module.

        Returns:
            torch.Tensor: Projected representations.
        """
        return self.projection_head(z)

    def normalize_with_temp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given tensor with the temperature.

        Args:
            x (torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return torch.exp(x / self.tau)

    def reset_parameters(self) -> None:
        """Reset the parameters of the model."""
        for conv_layer in self.encoder_module.conv:
            conv_layer.reset_parameters()
        for layer in self.projection_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()




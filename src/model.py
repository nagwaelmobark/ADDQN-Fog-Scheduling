"""
Neural-network architecture for ADDQN fog scheduling.

The model combines:
1. Shared node encoding
2. Learned node-wise attention
3. Global attention-weighted representation
4. Flattened node-level representation
5. Dual-path feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    """Shared encoder applied independently to every fog node."""

    def __init__(
        self,
        input_dim=5,
        hidden_dim=32,
        output_dim=16,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim,
        )
        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            output_dim,
        )
        self.bn2 = nn.BatchNorm1d(
            output_dim
        )

    def forward(self, x):
        original_shape = x.shape

        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, num_nodes, features = x.shape

        x = x.view(
            -1,
            features,
        )

        x = F.relu(
            self.bn1(
                self.fc1(x)
            )
        )

        x = F.relu(
            self.bn2(
                self.fc2(x)
            )
        )

        x = x.view(
            batch_size,
            num_nodes,
            -1,
        )

        if len(original_shape) == 2:
            x = x.squeeze(0)

        return x


class NodeAttention(nn.Module):
    """
    Learned node-wise attention.

    A scalar score is assigned to each node embedding.
    Softmax converts the scores into normalized node weights.
    """

    def __init__(
        self,
        embed_dim=16,
    ):
        super().__init__()

        self.attention = nn.Linear(
            embed_dim,
            1,
        )

    def forward(self, node_embeddings):
        scores = self.attention(
            node_embeddings
        )

        weights = F.softmax(
            scores,
            dim=1,
        )

        context = (
            node_embeddings * weights
        ).sum(dim=1)

        return context, weights


class ADDQN(nn.Module):
    """
    Attention-Enhanced Double Deep Q-Network architecture.

    The network uses two feature-processing paths:
    - an attention-weighted global path;
    - a flattened node-level path.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        num_nodes=15,
        hidden_dim=128,
    ):
        super().__init__()

        self.num_nodes = int(
            num_nodes
        )

        self.features_per_node = 5

        if state_dim != (
            self.num_nodes
            * self.features_per_node
        ):
            raise ValueError(
                "state_dim must equal num_nodes * 5."
            )

        # Shared node encoder: 5 -> 32 -> 16
        self.node_encoder = NodeEncoder(
            input_dim=self.features_per_node,
            hidden_dim=32,
            output_dim=16,
        )

        # Learned node-wise attention
        self.attention = NodeAttention(
            embed_dim=16
        )

        # Global attention path
        self.global_path = nn.Sequential(
            nn.Linear(
                16,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Node-level flattened path
        self.node_path = nn.Sequential(
            nn.Linear(
                16 * self.num_nodes,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Dual-path fusion
        self.combiner = nn.Linear(
            hidden_dim * 2,
            hidden_dim,
        )

        # Q-value output for all candidate fog nodes
        self.q_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        x = x.view(
            batch_size,
            self.num_nodes,
            self.features_per_node,
        )

        # Shared encoding for all nodes
        node_embeddings = self.node_encoder(
            x
        )

        # Path 1: attention-weighted global context
        context, _ = self.attention(
            node_embeddings
        )

        global_features = self.global_path(
            context
        )

        # Path 2: detailed node-level representation
        node_flat = node_embeddings.view(
            batch_size,
            -1,
        )

        node_features = self.node_path(
            node_flat
        )

        # Fuse both representations
        combined = torch.cat(
            [
                global_features,
                node_features,
            ],
            dim=1,
        )

        features = F.relu(
            self.combiner(
                combined
            )
        )

        q_values = self.q_head(
            features
        )

        return q_values
import torch
from torch import nn


class MechanismEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, mechanism_feature):
        return self.norm(self.net(mechanism_feature))


class ObservationPointEncoder(nn.Module):
    def __init__(self, hidden_dim, time_dim, num_vars, num_nodes=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_var_embeddings = max(int(num_vars), 1)
        self.use_node_embedding = num_nodes is not None and int(num_nodes) > 0
        self.num_node_embeddings = max(int(num_nodes or 1), 1)

        self.value_encoder = nn.Linear(1, hidden_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim),
        )
        self.var_embedding = nn.Embedding(self.num_var_embeddings, hidden_dim)
        self.mechanism_encoder = MechanismEncoder(hidden_dim)
        if self.use_node_embedding:
            self.node_embedding = nn.Embedding(self.num_node_embeddings, hidden_dim)
        else:
            self.node_embedding = None

        input_width = hidden_dim * (5 if self.use_node_embedding else 4)
        self.output_proj = nn.Linear(input_width, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _validate_var_ids(self, point_var_id):
        if point_var_id.numel() == 0:
            return
        min_var_id = int(point_var_id.min().item())
        max_var_id = int(point_var_id.max().item())
        if min_var_id < 0 or max_var_id >= self.num_var_embeddings:
            raise ValueError(
                f"Observed var_id range [{min_var_id}, {max_var_id}] exceeds configured num_vars={self.num_var_embeddings}."
            )

    def _validate_node_ids(self, point_node_id):
        if point_node_id.numel() == 0:
            return
        max_node_id = int(point_node_id.max().item())
        if max_node_id >= self.num_node_embeddings:
            raise ValueError(
                f"Observed node_id {max_node_id} exceeds configured num_nodes={self.num_node_embeddings}."
            )

    def forward(self, point_value, point_time, point_var_id, mechanism_feature=None, point_node_id=None):
        point_var_id = point_var_id.long()
        self._validate_var_ids(point_var_id)
        value_feature = self.value_encoder(point_value)
        time_feature = self.time_encoder(point_time)
        var_feature = self.var_embedding(point_var_id)
        if mechanism_feature is None:
            mechanism_feature = value_feature.new_zeros(value_feature.shape)
        else:
            mechanism_feature = self.mechanism_encoder(mechanism_feature)

        features = [value_feature, time_feature, var_feature, mechanism_feature]
        if self.use_node_embedding:
            if point_node_id is None:
                point_node_id = torch.zeros_like(point_var_id)
            point_node_id = point_node_id.long().clamp_min(0)
            self._validate_node_ids(point_node_id)
            features.append(self.node_embedding(point_node_id))

        point_feature = torch.cat(features, dim=-1)
        point_feature = self.output_proj(point_feature)
        return self.norm(point_feature)

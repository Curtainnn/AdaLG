import torch
from torch import nn

from layers.causal_ode import CausalODEFeaturePropagation
from layers.mechanism_encoder import ObservationPointEncoder
from layers.point_to_token import PointToTokenAggregator


class TimerXLPointFrontend(nn.Module):
    def __init__(
        self,
        input_token_len,
        hidden_dim,
        num_layers=0,
        knn=8,
        neighbor_window=-1,
        token_subbins=4,
        time_dim=32,
        point_use_topology=False,
        num_vars=1,
        num_nodes=None,
        causal_ode_state_dim=32,
        causal_ode_hidden=128,
        causal_ode_constraint_strength=0.0,
        causal_ode_h_weight=1e-3,
        causal_ode_l1_weight=1e-4,
        causal_ode_diag_weight=1e-3,
        causal_ode_laplacian_weight=1e-4,
        causal_ode_adjacency_scale=1.0,
        causal_ode_gate_init=-4.0,
        causal_ode_steps=8,
        causal_prior_window=30,
    ):
        super().__init__()
        self.input_token_len = input_token_len
        self.mechanism_density_window = float(input_token_len)
        self.encoder = ObservationPointEncoder(hidden_dim, time_dim, num_vars, num_nodes=num_nodes)
        self.causal_ode = CausalODEFeaturePropagation(
            hidden_dim=hidden_dim,
            num_vars=int(num_vars),
            state_dim=int(causal_ode_state_dim),
            ode_hidden=int(causal_ode_hidden),
            constraint_strength=float(causal_ode_constraint_strength),
            h_weight=float(causal_ode_h_weight),
            l1_weight=float(causal_ode_l1_weight),
            diag_weight=float(causal_ode_diag_weight),
            laplacian_weight=float(causal_ode_laplacian_weight),
            adjacency_scale=float(causal_ode_adjacency_scale),
            gate_init=float(causal_ode_gate_init),
            ode_steps=int(causal_ode_steps),
            dropout=0.0,
            causal_prior_window=int(causal_prior_window),
        )
        self.aggregator = PointToTokenAggregator(hidden_dim=hidden_dim, num_subbins=token_subbins)

    def get_regularization_loss(self):
        return self.causal_ode.get_regularization_loss()

    def get_causal_ode_gate(self):
        return self.causal_ode.get_residual_gate_value()

    def _prepare_dense_inputs(self, x, x_mask):
        if x_mask is None:
            x_mask = torch.isfinite(x)
        else:
            if x_mask.dim() == 2:
                x_mask = x_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            elif x_mask.dim() == 3 and x_mask.shape[-1] == 1:
                x_mask = x_mask.expand(-1, -1, x.shape[-1])
            elif x_mask.shape != x.shape:
                raise ValueError("x_mask must match x shape or broadcast across variable dimension.")
            x_mask = x_mask.to(dtype=torch.bool)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, x_mask

    def _build_mechanism_features(self, obs_time, obs_var_id, obs_batch_id, batch_size, num_vars, seq_len):
        device = obs_time.device
        num_points = obs_time.shape[0]
        mechanism_feature = torch.zeros(num_points, 4, dtype=obs_time.dtype, device=device)
        if num_points == 0:
            return mechanism_feature

        time_denom = float(max(int(seq_len) - 1, 1))
        normalized_time = obs_time.to(dtype=mechanism_feature.dtype) / time_denom
        density_window = max(self.mechanism_density_window / time_denom, 1e-6)

        batch_ids = obs_batch_id.unique(sorted=True).detach().cpu().tolist()
        for batch_id in batch_ids:
            local_mask = obs_batch_id == batch_id
            local_indices = local_mask.nonzero(as_tuple=False).squeeze(-1)
            if local_indices.numel() == 0:
                continue

            local_time = normalized_time[local_indices]
            time_distance = torch.abs(local_time.unsqueeze(1) - local_time.unsqueeze(0))
            mechanism_feature[local_indices, 1] = (
                (time_distance <= density_window).float().sum(dim=1) / (2.0 * density_window + 1e-6)
            )

            local_var_id = obs_var_id[local_indices].long()
            for var_id in range(int(num_vars)):
                var_mask = local_var_id == var_id
                var_indices = local_indices[var_mask]
                if var_indices.numel() == 0:
                    continue

                var_time = normalized_time[var_indices]
                sort_order = torch.argsort(var_time)
                sorted_indices = var_indices[sort_order]
                sorted_time = var_time[sort_order]

                delta_t = torch.zeros_like(sorted_time)
                if sorted_time.numel() > 1:
                    delta_t[1:] = sorted_time[1:] - sorted_time[:-1]
                    delta_t[0] = delta_t[1]
                    avg_dt = delta_t[1:].mean()
                else:
                    delta_t.fill_(1.0)
                    avg_dt = delta_t[0]

                mechanism_feature[sorted_indices, 0] = delta_t
                mechanism_feature[sorted_indices, 2] = 1.0 / (avg_dt + 1e-6)
                mechanism_feature[sorted_indices, 3] = 1.0 - (
                    float(sorted_indices.numel()) / float(max(int(seq_len), 1))
                )

        return mechanism_feature

    def _encode_points(
        self,
        obs_value,
        obs_time,
        obs_var_id,
        obs_batch_id,
        batch_size,
        num_vars,
        seq_len,
        obs_node_id=None,
    ):
        point_value = obs_value.unsqueeze(-1)
        time_denom = max(seq_len - 1, 1)
        point_time = obs_time.to(obs_value.dtype).unsqueeze(-1) / time_denom
        mechanism_feature = self._build_mechanism_features(
            obs_time=obs_time,
            obs_var_id=obs_var_id.long(),
            obs_batch_id=obs_batch_id.long(),
            batch_size=batch_size,
            num_vars=num_vars,
            seq_len=seq_len,
        )

        point_feature = self.encoder(
            point_value=point_value,
            point_time=point_time,
            point_var_id=obs_var_id.long(),
            mechanism_feature=mechanism_feature,
            point_node_id=obs_node_id.long() if obs_node_id is not None else None,
        )
        point_feature, causal_context_raw, causal_prior = self.causal_ode(
            point_feature=point_feature,
            batch_idx=obs_batch_id.long(),
            time_idx=obs_time,
            var_idx=obs_var_id.long(),
            batch_size=batch_size,
            num_vars=num_vars,
            seq_len=seq_len,
            return_context=True,
        )
        token_feature, token_mask, n_tokens = self.aggregator(
            point_feature=point_feature,
            batch_idx=obs_batch_id.long(),
            time_idx=obs_time,
            var_idx=obs_var_id.long(),
            input_token_len=self.input_token_len,
            batch_size=batch_size,
            num_vars=num_vars,
            seq_len=seq_len,
        )
        return token_feature, token_mask, n_tokens, causal_context_raw, causal_prior

    def forward(self, x, x_mark=None, x_mask=None, seq_len=None):
        if isinstance(x, dict):
            obs_node_id = x.get("obs_node_id")
            return self._encode_points(
                obs_value=x["obs_value"],
                obs_time=x["obs_time"],
                obs_var_id=x["obs_var_id"],
                obs_batch_id=x["obs_batch_id"],
                batch_size=int(x["batch_size"]),
                num_vars=int(x["num_vars"]),
                seq_len=int(x["obs_seq_len"]),
                obs_node_id=obs_node_id,
            )

        batch_size, _, num_vars = x.shape
        x, x_mask = self._prepare_dense_inputs(x, x_mask)
        if x_mark is None:
            x_mark = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).view(1, -1, 1).expand(batch_size, -1, -1)
        elif x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(-1)
        elif x_mark.dim() != 3 or x_mark.shape[-1] != 1:
            raise ValueError("x_mark must have shape [B, L, 1] in dense mode.")
        valid_idx = x_mask.nonzero(as_tuple=False)
        obs_batch_id = valid_idx[:, 0]
        obs_time = x_mark[obs_batch_id, valid_idx[:, 1], 0].to(torch.float32)
        obs_var_id = valid_idx[:, 2]
        obs_value = x[obs_batch_id, valid_idx[:, 1], obs_var_id]
        return self._encode_points(
            obs_value=obs_value,
            obs_time=obs_time,
            obs_var_id=obs_var_id,
            obs_batch_id=obs_batch_id,
            batch_size=batch_size,
            num_vars=num_vars,
            seq_len=int(seq_len) if seq_len is not None else (int(torch.ceil(x_mark[..., 0].max()).item()) + 1 if x_mark.numel() > 0 else x.shape[1]),
        )

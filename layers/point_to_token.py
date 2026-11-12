import torch
from torch import nn


class PointToTokenAggregator(nn.Module):
    def __init__(self, hidden_dim, num_subbins=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_subbins = max(int(num_subbins), 1)
        self.subbin_embedding = nn.Embedding(self.num_subbins, hidden_dim)
        fusion_width = hidden_dim * (self.num_subbins + 1) + self.num_subbins
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_width, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        point_feature,
        batch_idx,
        time_idx,
        var_idx,
        input_token_len,
        batch_size,
        num_vars,
        seq_len,
    ):
        if seq_len <= 0 or input_token_len <= 0:
            raise ValueError("History length and token width must be positive.")
        # The last region may be partial and includes the history endpoint.
        n_tokens = (seq_len + input_token_len - 1) // input_token_len

        token_feature = point_feature.new_zeros(batch_size, num_vars, n_tokens, self.hidden_dim)
        token_mask = torch.zeros(batch_size, num_vars, n_tokens, dtype=torch.bool, device=point_feature.device)

        if point_feature.shape[0] == 0:
            return token_feature, token_mask, n_tokens

        if not torch.isfinite(time_idx).all() or ((time_idx < 0) | (time_idx > seq_len)).any():
            raise ValueError("Observation timestamps must lie in the closed history interval [0, seq_len].")
        # Quantize only the pooling assignment. Keep original times for encoding
        # and Recent-K. The closed right endpoint belongs to the last subbin.
        assignment_time = time_idx.long().clamp(max=seq_len - 1)
        token_idx = torch.div(assignment_time, input_token_len, rounding_mode="floor")
        region_start = token_idx * input_token_len
        region_width = (seq_len - region_start).clamp(max=input_token_len)
        point_offset = assignment_time - region_start
        subbin_idx = torch.div(
            point_offset * self.num_subbins,
            region_width,
            rounding_mode="floor",
        ).clamp(max=self.num_subbins - 1)
        subbin_idx = torch.where(time_idx == seq_len, self.num_subbins - 1, subbin_idx)

        flat_subbin_size = batch_size * num_vars * n_tokens * self.num_subbins
        subbin_feature = point_feature.new_zeros(flat_subbin_size, self.hidden_dim)
        subbin_count = point_feature.new_zeros(flat_subbin_size, 1)

        flat_subbin_idx = (
            ((batch_idx * num_vars + var_idx) * n_tokens + token_idx) * self.num_subbins + subbin_idx
        )
        subbin_feature.index_add_(0, flat_subbin_idx, point_feature)
        subbin_count.index_add_(0, flat_subbin_idx, point_feature.new_ones(flat_subbin_idx.shape[0], 1))
        subbin_feature = subbin_feature / subbin_count.clamp_min(1.0)

        subbin_feature = subbin_feature.view(batch_size, num_vars, n_tokens, self.num_subbins, self.hidden_dim)
        subbin_count = subbin_count.view(batch_size, num_vars, n_tokens, self.num_subbins, 1)
        subbin_mask = (subbin_count > 0).to(subbin_feature.dtype)

        subbin_pos = self.subbin_embedding.weight.view(1, 1, 1, self.num_subbins, self.hidden_dim)
        subbin_feature = subbin_feature + subbin_pos * subbin_mask

        token_count = subbin_count.sum(dim=3)
        safe_token_count = token_count.clamp_min(1.0)
        has_token = (token_count > 0).to(subbin_feature.dtype)
        global_feature = (subbin_feature * subbin_count).sum(dim=3) / safe_token_count
        subbin_density = subbin_count.squeeze(-1) / safe_token_count.squeeze(-1).unsqueeze(-1)

        fusion_input = torch.cat(
            [
                subbin_feature.reshape(batch_size, num_vars, n_tokens, self.num_subbins * self.hidden_dim),
                global_feature,
                subbin_density,
            ],
            dim=-1,
        )
        token_feature = self.fusion_proj(fusion_input)
        token_feature = self.norm(token_feature) * has_token
        token_mask = token_count.squeeze(-1) > 0
        return token_feature, token_mask, n_tokens


TemporalTokenAggregator = PointToTokenAggregator

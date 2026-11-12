import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = 'grafiti'
MODEL_ALIASES = ('GraFITi', 'GRAFITI')
SUPPORTS_MASK_ARGS = True


def _batch_flatten(x_list, mask):
    batch_size, time_len, num_vars = x_list[0].shape
    flat_mask = mask.reshape(batch_size, time_len * num_vars).to(dtype=torch.bool)
    observed_counts = flat_mask.sum(dim=1)
    max_count = int(observed_counts.max().detach().cpu().item()) if observed_counts.numel() else 0
    max_count = max(max_count, 1)
    row_positions = torch.arange(max_count, device=mask.device).expand(batch_size, max_count)
    valid_positions = row_positions < observed_counts.unsqueeze(1)

    padded_values = []
    for tensor in x_list:
        flat_tensor = tensor.reshape(batch_size, time_len * num_vars)
        selected_values = flat_tensor[flat_mask]
        padded = tensor.new_zeros(batch_size, max_count)
        padded[valid_positions] = selected_values
        padded_values.append(padded)
    return padded_values


def _gather_nodes(nodes, indices):
    return nodes.gather(1, indices[:, :, None].expand(-1, -1, nodes.shape[-1]))


def _reconstruct_target(target_mask, flat_output, flat_target_mask):
    reconstructed = flat_output.new_zeros(target_mask.shape)
    target_indices = torch.nonzero(target_mask.to(dtype=torch.bool), as_tuple=True)
    reconstructed[target_indices] = flat_output[flat_target_mask.to(dtype=torch.bool)]
    return reconstructed


class MAB2(nn.Module):
    def __init__(self, dim_q, dim_k, hidden_dim, num_heads):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError('GraFITi latent dimension must be divisible by attention heads.')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(int(dim_q), hidden_dim)
        self.fc_k = nn.Linear(int(dim_k), hidden_dim)
        self.fc_v = nn.Linear(int(dim_k), hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, mask=None):
        query = self.fc_q(query)
        key, value = self.fc_k(key), self.fc_v(key)

        head_dim = self.hidden_dim // self.num_heads
        query_heads = torch.cat(query.split(head_dim, dim=2), dim=0)
        key_heads = torch.cat(key.split(head_dim, dim=2), dim=0)
        value_heads = torch.cat(value.split(head_dim, dim=2), dim=0)

        scores = query_heads.bmm(key_heads.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.repeat(self.num_heads, 1, 1) == 0, -1e10)
        attention = torch.softmax(scores, dim=2)
        output = torch.cat((query_heads + attention.bmm(value_heads)).split(query.size(0), dim=0), dim=2)
        return output + F.relu(self.fc_o(output))


class GraFITiCore(nn.Module):
    def __init__(self, num_vars, latent_dim=128, n_layers=2, attn_heads=4):
        super().__init__()
        self.num_vars = int(num_vars)
        self.latent_dim = int(latent_dim)
        self.n_layers = int(n_layers)
        self.attn_heads = int(attn_heads)

        self.edge_init = nn.Linear(2, self.latent_dim)
        self.channel_init = nn.Linear(self.num_vars, self.latent_dim)
        self.time_init = nn.Linear(1, self.latent_dim)

        self.channel_time_attn = nn.ModuleList(
            [
                MAB2(self.latent_dim, 2 * self.latent_dim, self.latent_dim, self.attn_heads)
                for _ in range(self.n_layers)
            ]
        )
        self.time_channel_attn = nn.ModuleList(
            [
                MAB2(self.latent_dim, 2 * self.latent_dim, self.latent_dim, self.attn_heads)
                for _ in range(self.n_layers)
            ]
        )
        self.edge_nn = nn.ModuleList([nn.Linear(3 * self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.output = nn.Linear(3 * self.latent_dim, 1)

    def _one_hot_channels(self, batch_size, device):
        indices = torch.arange(self.num_vars, device=device).expand(batch_size, self.num_vars)
        return F.one_hot(indices, num_classes=self.num_vars).float()

    def _build_indices(self, time_points):
        batch_size, time_len = time_points.shape
        time_indices = torch.arange(time_len, device=time_points.device).expand(
            batch_size,
            self.num_vars,
            time_len,
        ).permute(0, 2, 1)
        channel_indices = torch.arange(self.num_vars, device=time_points.device).expand(batch_size, time_len, -1)
        return time_indices, channel_indices

    def _create_attention_masks(self, mask, time_indices, channel_indices, time_points):
        batch_size, time_len = time_points.shape
        channels = torch.arange(self.num_vars, device=time_points.device).expand(batch_size, self.num_vars)
        channel_mask = (channels[:, :, None] == channel_indices[:, None, :]).float() * mask[:, None, :]
        time_steps = torch.arange(time_len, device=time_points.device)[None, :, None]
        time_mask = (time_indices[:, None, :] == time_steps).float() * mask[:, None, :]
        return time_mask, channel_mask

    def forward(self, time_points, values, obs_mask, target_mask):
        batch_size, _, num_vars = values.shape
        if num_vars != self.num_vars:
            raise ValueError(f'GraFITi expected {self.num_vars} variables, got {num_vars}.')

        time_nodes = time_points.unsqueeze(-1)
        channel_nodes = self._one_hot_channels(batch_size, device=time_points.device)
        time_indices, channel_indices = self._build_indices(time_points)

        edge_mask = (obs_mask | target_mask).to(dtype=torch.bool)
        flat_values = _batch_flatten(
            [time_indices, values, target_mask.float(), channel_indices, edge_mask],
            edge_mask,
        )
        time_indices_flat, obs_values, target_mask_flat, channel_indices_flat, flat_mask = flat_values
        time_indices_flat = time_indices_flat.to(dtype=torch.long)
        channel_indices_flat = channel_indices_flat.to(dtype=torch.long)
        flat_mask = flat_mask.to(dtype=torch.bool)

        target_indicator = (1.0 - flat_mask.float()) + target_mask_flat.float()
        edge_input = torch.cat([obs_values.unsqueeze(-1), target_indicator.unsqueeze(-1)], dim=-1)
        time_mask, channel_mask = self._create_attention_masks(
            flat_mask.float(),
            time_indices_flat,
            channel_indices_flat,
            time_points,
        )

        edge_emb = F.relu(self.edge_init(edge_input)) * flat_mask[:, :, None].float()
        time_emb = torch.sin(self.time_init(time_nodes))
        channel_emb = F.relu(self.channel_init(channel_nodes))

        for layer in range(self.n_layers):
            gathered_time = _gather_nodes(time_emb, time_indices_flat)
            gathered_channel = _gather_nodes(channel_emb, channel_indices_flat)

            channel_emb = self.channel_time_attn[layer](
                channel_emb,
                torch.cat([gathered_time, edge_emb], dim=-1),
                channel_mask,
            )
            time_emb = self.time_channel_attn[layer](
                time_emb,
                torch.cat([gathered_channel, edge_emb], dim=-1),
                time_mask,
            )

            edge_update = torch.cat([edge_emb, gathered_time, gathered_channel], dim=-1)
            edge_emb = F.relu(edge_emb + self.edge_nn[layer](edge_update)) * flat_mask[:, :, None].float()

        gathered_time = _gather_nodes(time_emb, time_indices_flat)
        gathered_channel = _gather_nodes(channel_emb, channel_indices_flat)
        flat_output = self.output(torch.cat([edge_emb, gathered_time, gathered_channel], dim=-1)).squeeze(-1)
        return _reconstruct_target(target_mask, flat_output, target_mask_flat)


class Model(nn.Module):
    supports_mask_args = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_vars = int(getattr(configs, 'enc_in', getattr(configs, 'n_vars', 1)))
        self.features = str(getattr(configs, 'features', 'M')).upper()
        self.use_norm = bool(getattr(configs, 'use_norm', False))
        self.core = GraFITiCore(
            num_vars=self.num_vars,
            latent_dim=int(getattr(configs, 'grafiti_latent_dim', getattr(configs, 'd_model', 128))),
            n_layers=int(getattr(configs, 'grafiti_n_layers', 2)),
            attn_heads=int(getattr(configs, 'grafiti_attn_heads', 4)),
        )

    def _prepare_time_mark(self, values, mark, start=0.0):
        batch_size, length = values.shape[0], values.shape[1]
        if mark is None:
            mark = torch.arange(length, device=values.device, dtype=values.dtype).view(1, length, 1) + float(start)
            return mark.expand(batch_size, -1, -1)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        if mark.dim() != 3 or mark.shape[0] != batch_size or mark.shape[1] != length:
            raise ValueError('GraFITi time mark must have shape [B, L] or [B, L, 1].')
        return mark.to(device=values.device, dtype=values.dtype)

    def _prepare_mask(self, values, mask):
        if mask is None:
            return torch.isfinite(values)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, values.shape[-1])
        elif mask.shape != values.shape:
            raise ValueError('GraFITi mask must match values or broadcast over variables.')
        return mask.to(device=values.device, dtype=torch.bool)

    def _normalize_history(self, x, x_mask):
        if not self.use_norm:
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), None, None
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask_float = x_mask.to(dtype=safe_x.dtype)
        valid_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        means = (safe_x * mask_float).sum(dim=1, keepdim=True) / valid_count
        centered = torch.where(x_mask, safe_x - means, torch.zeros_like(safe_x))
        variance = centered.pow(2).sum(dim=1, keepdim=True) / valid_count
        stdev = torch.sqrt(variance + 1e-5)
        return centered / stdev, means.detach(), stdev

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None, y=None, **kwargs):
        del kwargs
        if x.dim() != 3:
            raise ValueError('GraFITi expects x with shape [B, L, C].')
        batch_size, history_len, num_vars = x.shape
        if num_vars != self.num_vars:
            raise ValueError(f'GraFITi expected {self.num_vars} variables, got {num_vars}.')

        x_mask = self._prepare_mask(x, x_mask)
        x_mark = self._prepare_time_mark(x, x_mark)
        future_len = int(y_mark.shape[1]) if y_mark is not None else int(getattr(self.configs, 'output_token_len', 1))
        if y is not None:
            future_len = int(y.shape[1])

        future_reference = x.new_zeros(batch_size, future_len, num_vars)
        y_mark = self._prepare_time_mark(future_reference, y_mark, start=history_len)
        if y_mask is None:
            y_mask = torch.ones(batch_size, future_len, num_vars, device=x.device, dtype=torch.bool)
            if y is not None:
                y_mask = torch.isfinite(y)
        else:
            y_mask = self._prepare_mask(future_reference, y_mask)

        x, means, stdev = self._normalize_history(x, x_mask)
        time_points = torch.cat([x_mark[:, :, 0], y_mark[:, :, 0]], dim=1)
        time_min = time_points.amin(dim=1, keepdim=True)
        time_max = time_points.amax(dim=1, keepdim=True)
        time_points = (time_points - time_min) / (time_max - time_min).clamp_min(1e-6)

        future_values = x.new_zeros(batch_size, future_len, num_vars)
        values = torch.cat([x, future_values], dim=1)
        obs_mask = torch.cat([x_mask, torch.zeros_like(y_mask)], dim=1)
        target_mask = torch.cat([torch.zeros_like(x_mask), y_mask], dim=1)
        pred = self.core(time_points, values, obs_mask, target_mask)[:, history_len:, :]

        if self.use_norm:
            pred = pred * stdev + means

        if self.features == 'MS':
            pred = pred[:, :, -1:]
            if y is not None:
                y = y[:, :, -1:]
                y_mask = y_mask[:, :, -1:]

        if y is None:
            return pred
        return {'pred': pred, 'true': y, 'mask': y_mask}

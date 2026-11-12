import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder as FormerEncoder
from layers.Transformer_EncDec import EncoderLayer as FormerEncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention


MODEL_NAME = 'timecheat'
MODEL_ALIASES = ('TimeCHEAT', 'time_cheat', 'time-cheat')
SUPPORTS_MASK_ARGS = True


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f'TimeCHEAT patch count {x.size(1)} exceeds positional max_len={self.pe.size(1)}.')
        return self.pe[:, :x.size(1)].to(device=x.device, dtype=x.dtype)


class MAB2(nn.Module):
    def __init__(self, dim_q, dim_k, hidden_dim, num_heads):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError('TimeCHEAT latent_dim must be divisible by attn_heads.')
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


class GraphPatchEncoder(nn.Module):
    def __init__(self, num_vars, latent_dim=128, n_patches=8, n_layers=2, attn_heads=4):
        super().__init__()
        self.num_vars = int(num_vars)
        self.latent_dim = int(latent_dim)
        self.n_layers = int(n_layers)
        self.attn_heads = int(attn_heads)
        self.edge_init = nn.Linear(2, self.latent_dim)
        self.channel_init = nn.Linear(self.num_vars, self.latent_dim)
        self.time_init = nn.Linear(1, self.latent_dim)
        self.channel_time_attn = nn.ModuleList()
        self.time_channel_attn = nn.ModuleList()
        self.edge_nn = nn.ModuleList()
        self.channel_attn = nn.ModuleList()
        self.output = nn.Linear(3 * self.latent_dim, 1)
        self.corr = nn.Parameter(torch.eye(self.num_vars).unsqueeze(0).repeat(int(n_patches), 1, 1))

        for _ in range(self.n_layers):
            self.channel_time_attn.append(MAB2(self.latent_dim, 2 * self.latent_dim, self.latent_dim, self.attn_heads))
            self.time_channel_attn.append(MAB2(self.latent_dim, 2 * self.latent_dim, self.latent_dim, self.attn_heads))
            self.edge_nn.append(nn.Linear(3 * self.latent_dim, self.latent_dim))
            self.channel_attn.append(MAB2(self.latent_dim, self.latent_dim, self.latent_dim, self.attn_heads))

    def _gather(self, x, indices):
        return x.gather(1, indices[:, :, None].expand(-1, -1, x.shape[-1]))

    def _normal_corr(self, batch_size, patch_id, eps=1e-6):
        corr = (self.corr[int(patch_id)] + self.corr[int(patch_id)].T) / 2.0
        corr_min, corr_max = torch.min(corr), torch.max(corr)
        corr = (corr - corr_min) / (corr_max - corr_min + eps)
        return corr.unsqueeze(0).repeat(batch_size, 1, 1)

    def _pad_flat(self, value, full_len):
        return F.pad(value, [0, int(full_len) - int(value.numel())], value=0)

    def forward(self, context_time, value, mask, target_mask, patch_id=0):
        num_vars = value.shape[-1]
        time_nodes = context_time[:, :, None]
        channel_ids = torch.arange(num_vars, device=value.device).view(1, num_vars).expand(value.shape[0], -1)
        time_indices = torch.arange(value.shape[1], device=value.device).view(1, -1, 1).expand_as(value)
        channel_indices = torch.arange(num_vars, device=value.device).view(1, 1, -1).expand_as(value)
        mask_bool = mask.to(dtype=torch.bool)
        full_len = int(mask.sum((1, 2)).max().detach().cpu().item())
        full_len = max(full_len, 1)

        flat_time_indices = torch.stack(
            [self._pad_flat(row[row_mask], full_len) for row, row_mask in zip(time_indices, mask_bool)]
        ).long()
        flat_values = torch.stack(
            [self._pad_flat(row[row_mask], full_len) for row, row_mask in zip(value, mask_bool)]
        )
        flat_target_mask = torch.stack(
            [self._pad_flat(row[row_mask], full_len) for row, row_mask in zip(target_mask, mask_bool)]
        )
        flat_channel_indices = torch.stack(
            [self._pad_flat(row[row_mask], full_len) for row, row_mask in zip(channel_indices, mask_bool)]
        ).long()
        flat_mask = torch.stack(
            [self._pad_flat(row[row_mask], full_len) for row, row_mask in zip(mask, mask_bool)]
        )

        channel_nodes = self._normal_corr(value.shape[0], patch_id).to(device=value.device, dtype=value.dtype)
        edge_indicator = 1.0 - flat_mask + flat_target_mask
        edge_input = torch.cat([flat_values[:, :, None], edge_indicator[:, :, None]], dim=-1)

        channel_mask = (channel_ids[:, :, None] == flat_channel_indices[:, None, :]).to(dtype=value.dtype)
        channel_mask = channel_mask * flat_mask[:, None, :]
        time_steps = torch.arange(value.shape[1], device=value.device).view(1, -1, 1)
        time_mask = (time_steps == flat_time_indices[:, None, :]).to(dtype=value.dtype)
        time_mask = time_mask * flat_mask[:, None, :]

        edge_emb = F.relu(self.edge_init(edge_input)) * flat_mask[:, :, None]
        time_emb = torch.sin(self.time_init(time_nodes))
        channel_emb = F.relu(self.channel_init(channel_nodes))

        for layer in range(self.n_layers):
            gathered_time = self._gather(time_emb, flat_time_indices)
            gathered_channel = self._gather(channel_emb, flat_channel_indices)
            channel_update = self.channel_time_attn[layer](
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
            edge_emb = F.relu(edge_emb + self.edge_nn[layer](edge_update)) * flat_mask[:, :, None]
            channel_emb = self.channel_attn[layer](channel_update, channel_update)

        gathered_time = self._gather(time_emb, flat_time_indices)
        gathered_channel = self._gather(channel_emb, flat_channel_indices)
        output = self.output(torch.cat([edge_emb, gathered_time, gathered_channel], dim=-1)).squeeze(-1)
        return output, flat_target_mask


class TimeCHEATBackbone(nn.Module):
    def __init__(
        self,
        num_vars,
        seq_len,
        pred_len,
        attn_heads=4,
        latent_dim=128,
        n_layers=2,
        ref_points=128,
        n_patches=8,
        dropout=0.1,
        former_dff=256,
        former_layers=3,
        former_heads=8,
        former_activation='gelu',
    ):
        super().__init__()
        self.num_vars = int(num_vars)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.n_patches = int(n_patches)
        self.ref_points_total = int(ref_points)
        if self.ref_points_total % self.n_patches != 0:
            raise ValueError('TimeCHEAT ref_points must be divisible by n_patches.')

        self.register_buffer('patch_range', torch.linspace(0, float(self.seq_len), self.n_patches + 1))
        reference_points = torch.linspace(0, float(self.seq_len), self.ref_points_total)
        self.ref_points_per_patch = self.ref_points_total // self.n_patches
        self.register_buffer('ref_points', reference_points.reshape(self.n_patches, self.ref_points_per_patch))

        self.encoder = GraphPatchEncoder(
            num_vars=self.num_vars,
            attn_heads=attn_heads,
            n_patches=self.n_patches,
            latent_dim=latent_dim,
            n_layers=n_layers,
        )
        self.position_embedding = PositionalEmbedding(self.ref_points_per_patch)
        self.dropout = nn.Dropout(float(dropout))
        self.former = FormerEncoder(
            [
                FormerEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=float(dropout), output_attention=False),
                        self.ref_points_per_patch,
                        int(former_heads),
                    ),
                    self.ref_points_per_patch,
                    int(former_dff),
                    dropout=float(dropout),
                    activation=str(former_activation),
                )
                for _ in range(int(former_layers))
            ],
            norm_layer=nn.LayerNorm(self.ref_points_per_patch),
        )
        self.linear = nn.Linear(self.n_patches * self.ref_points_per_patch, self.pred_len)

    def _split_patch(self, values, mask, time, patch_id):
        start = self.patch_range[patch_id].to(device=values.device, dtype=time.dtype)
        end = self.patch_range[patch_id + 1].to(device=values.device, dtype=time.dtype)
        ref_points = self.ref_points[patch_id].to(device=values.device, dtype=time.dtype)
        if patch_id == self.n_patches - 1:
            time_mask = (time >= start) & (time <= end) & (mask.sum(-1) > 0)
        else:
            time_mask = (time >= start) & (time < end) & (mask.sum(-1) > 0)
        num_observed = time_mask.sum(1).long()

        n_ref_points = int(ref_points.size(0))
        max_observed = int(num_observed.max().detach().cpu().item()) if num_observed.numel() else 0
        patch_len = max_observed + n_ref_points
        patch = values.new_zeros(values.size(0), patch_len, self.num_vars)
        patch_mask = values.new_zeros(values.size(0), patch_len, self.num_vars)
        patch_time = values.new_zeros(values.size(0), patch_len)
        ref_mask = values.new_zeros(values.size(0), patch_len, self.num_vars)
        indices = torch.arange(patch_len, device=values.device)

        for batch_id in range(values.size(0)):
            observed_count = int(num_observed[batch_id].item())
            if observed_count > 0:
                patch_mask[batch_id, :observed_count, :] = mask[batch_id, time_mask[batch_id]]
                patch[batch_id, :observed_count, :] = values[batch_id, time_mask[batch_id]]
                patch_time[batch_id, :observed_count] = time[batch_id, time_mask[batch_id]]
            patch_time[batch_id, observed_count:observed_count + n_ref_points] = ref_points
            sorted_index = torch.cat(
                [
                    torch.argsort(patch_time[batch_id, :observed_count + n_ref_points]),
                    indices[observed_count + n_ref_points:],
                ]
            )
            patch_mask[batch_id] = patch_mask[batch_id, sorted_index]
            patch[batch_id] = patch[batch_id, sorted_index]
            patch_time[batch_id] = patch_time[batch_id, sorted_index]
            ref_mask[batch_id, sorted_index[observed_count:observed_count + n_ref_points]] = 1.0
        return patch, patch_mask, patch_time, ref_mask

    def embedding(self, data):
        values = data[..., :self.num_vars]
        mask = data[..., self.num_vars:-1]
        time = data[..., -1]

        patch_representations = []
        for patch_id in range(self.n_patches):
            patch_values, patch_mask, patch_time, ref_mask = self._split_patch(values, mask, time, patch_id)
            patch_values = patch_values * patch_mask
            context_mask = patch_mask + ref_mask
            encoded, encoded_mask = self.encoder(
                patch_time,
                patch_values,
                context_mask,
                ref_mask,
                patch_id=patch_id,
            )
            patch_repr = encoded[encoded_mask == 1].reshape(values.size(0), -1, self.num_vars).unsqueeze(1)
            patch_representations.append(patch_repr)

        patch_repr = torch.cat(patch_representations, dim=1).contiguous().permute(0, 3, 1, 2)
        patch_repr = patch_repr.reshape(patch_repr.shape[0] * patch_repr.shape[1], patch_repr.shape[2], patch_repr.shape[3])
        patch_repr = self.dropout(patch_repr + self.position_embedding(patch_repr))
        encoded_patch, _ = self.former(patch_repr)
        encoded_patch = encoded_patch.reshape(-1, self.num_vars, encoded_patch.shape[-2], encoded_patch.shape[-1])
        return encoded_patch.permute(0, 1, 3, 2)

    def forward(self, data):
        embedding = self.embedding(data)
        pred = self.dropout(self.linear(torch.flatten(embedding, start_dim=-2)))
        return pred.permute(0, 2, 1)


class Model(nn.Module):
    supports_mask_args = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(getattr(configs, 'seq_len', 1))
        self.pred_len = int(
            max(
                getattr(configs, 'pred_len', 0),
                getattr(configs, 'output_token_len', 0),
                getattr(configs, 'test_pred_len', 0),
            )
        )
        self.num_vars = int(getattr(configs, 'enc_in', getattr(configs, 'n_vars', 1)))
        self.features = str(getattr(configs, 'features', 'M')).upper()
        self.use_norm = bool(getattr(configs, 'use_norm', False))
        self.backbone = TimeCHEATBackbone(
            num_vars=self.num_vars,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            attn_heads=int(getattr(configs, 'timecheat_attn_heads', 4)),
            latent_dim=int(getattr(configs, 'timecheat_latent_dim', getattr(configs, 'd_model', 128))),
            n_layers=int(getattr(configs, 'timecheat_n_layers', 2)),
            ref_points=int(getattr(configs, 'timecheat_ref_points', 128)),
            n_patches=int(getattr(configs, 'timecheat_n_patches', 8)),
            dropout=float(getattr(configs, 'dropout', 0.1)),
            former_dff=int(getattr(configs, 'timecheat_former_dff', 256)),
            former_layers=int(getattr(configs, 'timecheat_former_layers', 3)),
            former_heads=int(getattr(configs, 'timecheat_former_heads', 8)),
            former_activation=str(getattr(configs, 'activation', 'gelu')),
        )

    def _prepare_time_mark(self, x, x_mark):
        if x_mark is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            return torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, seq_len, 1).expand(batch_size, -1, -1)
        if x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(-1)
        if x_mark.dim() != 3:
            raise ValueError('TimeCHEAT x_mark must have shape [B, L] or [B, L, 1].')
        return x_mark.to(device=x.device, dtype=x.dtype)

    def _prepare_mask(self, values, mask):
        if mask is None:
            return torch.isfinite(values)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, values.shape[-1])
        elif mask.shape != values.shape:
            raise ValueError('TimeCHEAT mask must match values or broadcast over variables.')
        return mask.to(device=values.device, dtype=torch.bool)

    def _normalize_history(self, x, x_mask):
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.use_norm:
            return safe_x, None, None
        mask_float = x_mask.to(dtype=safe_x.dtype)
        valid_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        means = (safe_x * mask_float).sum(dim=1, keepdim=True) / valid_count
        centered = torch.where(x_mask, safe_x - means, torch.zeros_like(safe_x))
        variance = centered.pow(2).sum(dim=1, keepdim=True) / valid_count
        stdev = torch.sqrt(variance + 1e-5)
        return centered / stdev, means.detach(), stdev

    def _align_prediction_to_targets(self, pred, y_mark, target_len):
        if y_mark is None:
            if target_len > pred.shape[1]:
                raise ValueError(f'TimeCHEAT produced {pred.shape[1]} steps but target requires {target_len}.')
            return pred[:, :target_len, :]
        if y_mark.dim() == 3:
            target_time = y_mark[:, :, 0]
        elif y_mark.dim() == 2:
            target_time = y_mark
        else:
            raise ValueError('TimeCHEAT y_mark must have shape [B, T] or [B, T, 1].')
        horizon_slot = torch.ceil(target_time.to(dtype=pred.dtype) - float(self.seq_len)).long() - 1
        horizon_slot = horizon_slot.clamp(min=0, max=pred.shape[1] - 1)
        gather_index = horizon_slot.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
        return pred.gather(dim=1, index=gather_index)

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None, y=None, **kwargs):
        del kwargs
        x_mask = self._prepare_mask(x, x_mask)
        x_mark = self._prepare_time_mark(x, x_mark)
        x, means, stdev = self._normalize_history(x, x_mask)
        data = torch.cat([x * x_mask.to(dtype=x.dtype), x_mask.to(dtype=x.dtype), x_mark[:, :, :1]], dim=-1)
        pred = self.backbone(data)

        target_len = None
        if y is not None:
            target_len = int(y.shape[1])
        elif y_mark is not None:
            target_len = int(y_mark.shape[1])
        if target_len is not None:
            pred = self._align_prediction_to_targets(pred, y_mark, target_len)

        if self.use_norm:
            pred = pred * stdev + means

        if self.features == 'MS':
            pred = pred[:, :, -1:]
            if y is not None:
                y = y[:, :, -1:]
                y_mask = y_mask[:, :, -1:] if y_mask is not None else None

        if y is None:
            return pred
        if y_mask is None:
            y_mask = torch.isfinite(y)
        else:
            y_mask = y_mask.to(dtype=torch.bool)
        return {'pred': pred, 'true': y, 'mask': y_mask}

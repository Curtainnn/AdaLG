import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = 'tpatchgnn'
MODEL_ALIASES = ('tPatchGNN', 'TPatchGNN')
SUPPORTS_MASK_ARGS = True


class NConv(nn.Module):
    def forward(self, x, adjacency):
        return torch.einsum('bfnm,bmnv->bfvm', (x, adjacency)).contiguous()


class Linear2d(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=1):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear2d((int(order) * int(support_len) + 1) * c_in, c_out)
        self.dropout = float(dropout)
        self.order = int(order)

    def forward(self, x, support):
        out = [x]
        for adjacency in support:
            x1 = self.nconv(x, adjacency)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adjacency)
                out.append(x2)
                x1 = x2
        h = self.mlp(torch.cat(out, dim=1))
        return F.relu(F.dropout(h, p=self.dropout, training=self.training))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f'Patch count {x.size(1)} exceeds positional encoding length {self.pe.size(1)}.')
        return x + self.pe[:, :x.size(1), :].to(dtype=x.dtype, device=x.device)


class TPatchGNNCore(nn.Module):
    def __init__(
        self,
        num_vars,
        hid_dim=64,
        te_dim=10,
        node_dim=10,
        n_patches=1,
        n_heads=1,
        tf_layer=1,
        n_layer=1,
        hop=1,
        outlayer='Linear',
        dropout=0.1,
    ):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.num_vars = int(num_vars)
        self.n_patches = int(n_patches)
        self.n_layer = int(n_layer)
        self.node_dim = int(node_dim)
        self.outlayer = str(outlayer)

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, int(te_dim) - 1)

        input_dim = 1 + int(te_dim)
        ttcn_dim = self.hid_dim - 1
        self.ttcn_dim = ttcn_dim
        self.filter_generators = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, input_dim * ttcn_dim, bias=True),
        )
        self.t_bias = nn.Parameter(torch.randn(1, ttcn_dim))

        self.position = PositionalEncoding(self.hid_dim)
        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hid_dim,
                nhead=int(n_heads),
                batch_first=True,
                dropout=float(dropout),
            )
            self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=int(tf_layer)))

        self.nodevec1 = nn.Parameter(torch.randn(self.num_vars, self.node_dim))
        self.nodevec2 = nn.Parameter(torch.randn(self.node_dim, self.num_vars))
        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(self.n_layer):
            self.nodevec_linear1.append(nn.Linear(self.hid_dim, self.node_dim))
            self.nodevec_linear2.append(nn.Linear(self.hid_dim, self.node_dim))
            self.nodevec_gate1.append(nn.Sequential(nn.Linear(self.hid_dim + self.node_dim, 1), nn.Tanh(), nn.ReLU()))
            self.nodevec_gate2.append(nn.Sequential(nn.Linear(self.hid_dim + self.node_dim, 1), nn.Tanh(), nn.ReLU()))

        self.gconv = nn.ModuleList()
        for _ in range(self.n_layer):
            self.gconv.append(GCN(self.hid_dim, self.hid_dim, dropout, support_len=1, order=int(hop)))

        if self.outlayer == 'CNN':
            self.temporal_agg = nn.Sequential(nn.Conv1d(self.hid_dim, self.hid_dim, kernel_size=self.n_patches))
        elif self.outlayer == 'Linear':
            self.temporal_agg = nn.Sequential(nn.Linear(self.hid_dim * self.n_patches, self.hid_dim))
        else:
            raise ValueError("tPatchGNN outlayer must be 'Linear' or 'CNN'.")

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim + int(te_dim), self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, 1),
        )

    def learnable_time_embedding(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def ttcn(self, x_int, mask_x):
        num_series, seq_len, _ = mask_x.shape
        if seq_len == 0:
            return x_int.new_zeros(num_series, self.ttcn_dim)
        generated = self.filter_generators(x_int)
        generated = generated * mask_x + (1 - mask_x) * (-1e8)
        generated = F.softmax(generated, dim=-2)
        generated = generated.view(num_series, seq_len, self.ttcn_dim, -1)
        x_broad = x_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(x_broad * generated, dim=-3), dim=-1)
        return torch.relu(ttcn_out + self.t_bias)

    def encode_patches(self, x, mask_x, batch_size):
        mask_patch = (mask_x.sum(dim=1) > 0).to(dtype=x.dtype)
        x_patch = self.ttcn(x, mask_x)
        x_patch = torch.cat([x_patch, mask_patch], dim=-1)
        x_patch = x_patch.view(batch_size, self.num_vars, self.n_patches, -1)
        bsz, num_vars, n_patches, _ = x_patch.shape

        h = x_patch
        for layer in range(self.n_layer):
            if layer > 0:
                h_last = h

            h = h.reshape(bsz * num_vars, n_patches, -1)
            h = self.position(h)
            h = self.transformer_encoder[layer](h).view(x_patch.shape)

            nodevec1 = self.nodevec1.view(1, 1, num_vars, self.node_dim).repeat(bsz, n_patches, 1, 1)
            nodevec2 = self.nodevec2.view(1, 1, self.node_dim, num_vars).repeat(bsz, n_patches, 1, 1)
            gate1 = self.nodevec_gate1[layer](torch.cat([h, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            gate2 = self.nodevec_gate2[layer](torch.cat([h, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            h_p1 = gate1 * self.nodevec_linear1[layer](h)
            h_p2 = gate2 * self.nodevec_linear2[layer](h)
            nodevec1 = nodevec1 + h_p1.permute(0, 2, 1, 3)
            nodevec2 = nodevec2 + h_p2.permute(0, 2, 3, 1)

            adjacency = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1)
            h = self.gconv[layer](h.permute(0, 3, 1, 2), [adjacency]).permute(0, 2, 3, 1)

            if layer > 0:
                h = h_last + h

        if self.outlayer == 'CNN':
            h = h.reshape(batch_size * self.num_vars, self.n_patches, -1).permute(0, 2, 1)
            h = self.temporal_agg(h).view(batch_size, self.num_vars, -1)
        else:
            h = h.reshape(batch_size, self.num_vars, -1)
            h = self.temporal_agg(h)
        return h

    def forecasting(self, query_time, x, x_time, mask):
        bsz, _, _, num_vars = x.shape
        x = x.permute(0, 3, 1, 2).reshape(-1, x.shape[2], 1)
        x_time = x_time.permute(0, 3, 1, 2).reshape(-1, x_time.shape[2], 1)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, mask.shape[2], 1)
        x_time_embed = self.learnable_time_embedding(x_time)
        encoded_input = torch.cat([x, x_time_embed], dim=-1)
        h = self.encode_patches(encoded_input, mask, bsz)

        pred_len = query_time.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, pred_len, 1)
        query_time = query_time.view(bsz, 1, pred_len, 1).repeat(1, num_vars, 1, 1)
        query_time_embed = self.learnable_time_embedding(query_time)
        h = torch.cat([h, query_time_embed], dim=-1)
        return self.decoder(h).squeeze(dim=-1).permute(0, 2, 1)


class Model(nn.Module):
    supports_mask_args = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_vars = int(configs.enc_in)
        self.use_norm = bool(getattr(configs, 'use_norm', False))
        self.patch_size = float(getattr(configs, 'tpatch_patch_size', 0.0))
        if self.patch_size <= 0:
            self.patch_size = float(max(int(getattr(configs, 'input_token_len', 1)), 1))
        self.stride = float(getattr(configs, 'tpatch_stride', 0.0))
        if self.stride <= 0:
            self.stride = self.patch_size
        if self.patch_size <= 0 or self.stride <= 0:
            raise ValueError('tPatchGNN patch size and stride must be positive.')
        seq_len = int(getattr(configs, 'seq_len', 1))
        self.n_patches = int(math.ceil(max(seq_len - self.patch_size, 0.0) / self.stride)) + 1
        self.core = TPatchGNNCore(
            num_vars=self.num_vars,
            hid_dim=int(getattr(configs, 'tpatch_hid_dim', getattr(configs, 'd_model', 64))),
            te_dim=int(getattr(configs, 'tpatch_te_dim', 10)),
            node_dim=int(getattr(configs, 'tpatch_node_dim', 10)),
            n_patches=self.n_patches,
            n_heads=int(getattr(configs, 'tpatch_n_heads', 1)),
            tf_layer=int(getattr(configs, 'tpatch_tf_layer', 1)),
            n_layer=int(getattr(configs, 'tpatch_n_layer', 1)),
            hop=int(getattr(configs, 'tpatch_hop', 1)),
            outlayer=str(getattr(configs, 'tpatch_outlayer', 'Linear')),
            dropout=float(getattr(configs, 'dropout', 0.0)),
        )

    def _prepare_time_mark(self, x, x_mark):
        if x_mark is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            return torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, seq_len, 1).expand(batch_size, -1, -1)
        if x_mark.dim() == 2:
            return x_mark.unsqueeze(-1).to(dtype=x.dtype)
        return x_mark.to(dtype=x.dtype)

    def _prepare_mask(self, x, x_mask):
        if x_mask is None:
            return torch.isfinite(x)
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        elif x_mask.dim() == 3 and x_mask.shape[-1] == 1:
            x_mask = x_mask.expand(-1, -1, x.shape[-1])
        return x_mask.to(dtype=torch.bool)

    def _normalize_dense_input(self, x, x_mask=None):
        valid_mask = x_mask.to(dtype=torch.bool) if x_mask is not None else torch.isfinite(x)
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(safe_x.dtype)
        means = (safe_x * valid_mask.to(safe_x.dtype)).sum(dim=1, keepdim=True) / valid_count
        centered_x = torch.where(valid_mask, safe_x - means, torch.zeros_like(safe_x))
        variance = centered_x.pow(2).sum(dim=1, keepdim=True) / valid_count
        stdev = torch.sqrt(variance + 1e-5)
        return centered_x / stdev, means.detach(), stdev

    def _build_patch_tensors(self, x, x_mark, x_mask, time_denom):
        batch_size, _, num_vars = x.shape
        patch_indices = []
        max_patch_len = 1
        time_values = x_mark[:, :, 0]
        for batch_id in range(batch_size):
            for patch_id in range(self.n_patches):
                start = patch_id * self.stride
                end = start + self.patch_size
                if patch_id == self.n_patches - 1:
                    time_mask = (time_values[batch_id] >= start) & (time_values[batch_id] <= end)
                else:
                    time_mask = (time_values[batch_id] >= start) & (time_values[batch_id] < end)
                for var_id in range(num_vars):
                    indices = torch.nonzero(time_mask & x_mask[batch_id, :, var_id], as_tuple=False).squeeze(-1)
                    patch_indices.append((batch_id, patch_id, var_id, indices))
                    max_patch_len = max(max_patch_len, int(indices.numel()))

        patch_x = x.new_zeros(batch_size, self.n_patches, max_patch_len, num_vars)
        patch_time = x.new_zeros(batch_size, self.n_patches, max_patch_len, num_vars)
        patch_mask = x.new_zeros(batch_size, self.n_patches, max_patch_len, num_vars)
        for batch_id, patch_id, var_id, indices in patch_indices:
            count = int(indices.numel())
            if count == 0:
                continue
            patch_x[batch_id, patch_id, :count, var_id] = x[batch_id, indices, var_id]
            patch_time[batch_id, patch_id, :count, var_id] = x_mark[batch_id, indices, 0] / time_denom
            patch_mask[batch_id, patch_id, :count, var_id] = 1.0
        return patch_x, patch_time, patch_mask

    def forward(self, x, x_mark, y_mark, x_mask=None, y_mask=None, **kwargs):
        x_mask = self._prepare_mask(x, x_mask)
        x_mark = self._prepare_time_mark(x, x_mark)
        if y_mark.dim() == 2:
            y_mark = y_mark.unsqueeze(-1)
        y_mark = y_mark.to(dtype=x.dtype)

        if self.use_norm:
            x, means, stdev = self._normalize_dense_input(x, x_mask=x_mask)
        else:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            means = None
            stdev = None

        seq_len = int(getattr(self.configs, 'seq_len', x.shape[1]))
        pred_len = y_mark.shape[1]
        time_denom = float(max(seq_len + pred_len - 1, 1))
        patch_x, patch_time, patch_mask = self._build_patch_tensors(x, x_mark, x_mask, time_denom)
        query_time = y_mark[:, :, 0] / time_denom
        pred = self.core.forecasting(query_time, patch_x, patch_time, patch_mask)

        if self.use_norm:
            pred = pred * stdev + means
        if str(getattr(self.configs, 'features', 'M')).upper() == 'MS':
            return pred[:, :, -1:]
        return pred

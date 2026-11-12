import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = 'mtan'
MODEL_ALIASES = ('mTAN', 'Mtan', 'm-tan', 'm_tan')
SUPPORTS_MASK_ARGS = True


class MultiTimeAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, embed_time=16, num_heads=1):
        super().__init__()
        if int(embed_time) % int(num_heads) != 0:
            raise ValueError('mTAN embed_time must be divisible by mtan_num_heads.')
        self.embed_time = int(embed_time)
        self.embed_time_k = self.embed_time // int(num_heads)
        self.num_heads = int(num_heads)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.embed_time, self.embed_time),
                nn.Linear(self.embed_time, self.embed_time),
                nn.Linear(self.input_dim * self.num_heads, self.hidden_dim),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        value_dim = value.size(-1)
        key_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_dim)
        scores = scores.unsqueeze(-1).repeat_interleave(value_dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            attn = dropout(attn)
        return torch.sum(attn * value.unsqueeze(-3), dim=-2), attn

    def forward(self, query, key, value, mask=None, dropout=None):
        batch_size, _, value_dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            layer(tensor)
            .view(tensor.size(0), -1, self.num_heads, self.embed_time_k)
            .transpose(1, 2)
            for layer, tensor in zip(self.linears, (query, key))
        ]
        out, _ = self.attention(query, key, value, mask, dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * value_dim)
        return self.linears[-1](out)


class EncoderMTANRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=16,
        hidden_dim=64,
        embed_time=128,
        num_heads=1,
        learn_emb=True,
    ):
        super().__init__()
        self.embed_time = int(embed_time)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learn_emb = bool(learn_emb)
        self.register_buffer('query', query.float())
        self.att = MultiTimeAttention(2 * self.input_dim, self.hidden_dim, self.embed_time, num_heads)
        self.gru_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, int(latent_dim) * 2),
        )
        if self.learn_emb:
            self.periodic = nn.Linear(1, self.embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, time_steps):
        time_steps = time_steps.unsqueeze(-1)
        return torch.cat([self.linear(time_steps), torch.sin(self.periodic(time_steps))], dim=-1)

    def fixed_time_embedding(self, time_steps):
        pe = torch.zeros(time_steps.shape[0], time_steps.shape[1], self.embed_time, device=time_steps.device)
        position = 48.0 * time_steps.unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, self.embed_time, 2, device=time_steps.device) * -(np.log(10.0) / self.embed_time)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, values_with_mask, time_steps):
        mask = values_with_mask[:, :, self.input_dim:]
        mask = torch.cat([mask, mask], dim=2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0).to(device=time_steps.device, dtype=time_steps.dtype))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0).to(device=time_steps.device, dtype=time_steps.dtype))
        out = self.att(query, key, values_with_mask, mask)
        out, _ = self.gru_rnn(out)
        return self.hiddens_to_z0(out)


class DecoderMTANRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        query,
        latent_dim=16,
        hidden_dim=50,
        embed_time=128,
        num_heads=1,
        learn_emb=True,
    ):
        super().__init__()
        self.embed_time = int(embed_time)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learn_emb = bool(learn_emb)
        self.register_buffer('query', query.float())
        self.att = MultiTimeAttention(2 * self.hidden_dim, 2 * self.hidden_dim, self.embed_time, num_heads)
        self.gru_rnn = nn.GRU(int(latent_dim), self.hidden_dim, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, self.input_dim),
        )
        if self.learn_emb:
            self.periodic = nn.Linear(1, self.embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, time_steps):
        time_steps = time_steps.unsqueeze(-1)
        return torch.cat([self.linear(time_steps), torch.sin(self.periodic(time_steps))], dim=-1)

    def fixed_time_embedding(self, time_steps):
        pe = torch.zeros(time_steps.shape[0], time_steps.shape[1], self.embed_time, device=time_steps.device)
        position = 48.0 * time_steps.unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, self.embed_time, 2, device=time_steps.device) * -(np.log(10.0) / self.embed_time)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, latent, time_steps):
        out, _ = self.gru_rnn(latent)
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps)
            key = self.learn_time_embedding(self.query.unsqueeze(0).to(device=time_steps.device, dtype=time_steps.dtype))
        else:
            query = self.fixed_time_embedding(time_steps)
            key = self.fixed_time_embedding(self.query.unsqueeze(0).to(device=time_steps.device, dtype=time_steps.dtype))
        out = self.att(query, key, out)
        return self.z0_to_obs(out)


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
        self.latent_dim = int(getattr(configs, 'mtan_latent_dim', 16))
        self.k_samples = max(int(getattr(configs, 'mtan_k_samples', 1)), 1)
        self.kl_weight = float(getattr(configs, 'mtan_kl_weight', 0.0))
        self._regularization_loss = None

        num_ref_points = int(getattr(configs, 'mtan_num_ref_points', 64))
        query = torch.linspace(0.0, 1.0, num_ref_points)
        learn_emb = bool(getattr(configs, 'mtan_learn_emb', True))
        embed_time = int(getattr(configs, 'mtan_embed_time', 128))
        num_heads = int(getattr(configs, 'mtan_num_heads', getattr(configs, 'n_heads', 1)))
        self.enc = EncoderMTANRNN(
            input_dim=self.num_vars,
            query=query,
            latent_dim=self.latent_dim,
            hidden_dim=int(getattr(configs, 'mtan_rec_hidden', 64)),
            embed_time=embed_time,
            num_heads=num_heads,
            learn_emb=learn_emb,
        )
        self.dec = DecoderMTANRNN(
            input_dim=self.num_vars,
            query=query,
            latent_dim=self.latent_dim,
            hidden_dim=int(getattr(configs, 'mtan_gen_hidden', 50)),
            embed_time=embed_time,
            num_heads=num_heads,
            learn_emb=learn_emb,
        )

    def get_regularization_loss(self):
        return self._regularization_loss

    def _prepare_time_mark(self, values, mark, start=0.0):
        batch_size, length = values.shape[0], values.shape[1]
        if mark is None:
            mark = torch.arange(length, device=values.device, dtype=values.dtype).view(1, length, 1) + float(start)
            return mark.expand(batch_size, -1, -1)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        if mark.dim() != 3 or mark.shape[0] != batch_size or mark.shape[1] != length:
            raise ValueError('mTAN time mark must have shape [B, L] or [B, L, 1].')
        return mark.to(device=values.device, dtype=values.dtype)

    def _prepare_mask(self, values, mask):
        if mask is None:
            return torch.isfinite(values)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, values.shape[-1])
        elif mask.shape != values.shape:
            raise ValueError('mTAN mask must match values or broadcast over variables.')
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

    def _normalize_times(self, x_mark, y_mark):
        history_time = x_mark[:, :, 0]
        future_time = y_mark[:, :, 0]
        all_time = torch.cat([history_time, future_time], dim=1)
        time_min = all_time.amin(dim=1, keepdim=True)
        time_max = all_time.amax(dim=1, keepdim=True)
        denom = (time_max - time_min).clamp_min(1e-6)
        return (history_time - time_min) / denom, (future_time - time_min) / denom

    def _sample_latent(self, mean, logvar):
        if self.k_samples <= 1:
            return mean
        noise = torch.randn(
            self.k_samples,
            mean.shape[0],
            mean.shape[1],
            mean.shape[2],
            device=mean.device,
            dtype=mean.dtype,
        )
        latent = noise * torch.exp(0.5 * logvar).unsqueeze(0) + mean.unsqueeze(0)
        return latent.reshape(-1, mean.shape[1], mean.shape[2])

    def _decode_and_average(self, latent, future_time, batch_size):
        if self.k_samples <= 1:
            return self.dec(latent, future_time)
        repeated_time = future_time.unsqueeze(0).repeat(self.k_samples, 1, 1).reshape(-1, future_time.shape[1])
        pred = self.dec(latent, repeated_time)
        return pred.view(self.k_samples, batch_size, pred.shape[1], pred.shape[2]).mean(dim=0)

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None, y=None, **kwargs):
        del kwargs
        batch_size, _, num_vars = x.shape
        if num_vars != self.num_vars:
            raise ValueError(f'mTAN expected {self.num_vars} variables, got {num_vars}.')
        x_mask = self._prepare_mask(x, x_mask)
        x_mark = self._prepare_time_mark(x, x_mark)

        future_len = int(y.shape[1]) if y is not None else self.pred_len
        if y_mark is not None:
            future_len = int(y_mark.shape[1])
        future_reference = x.new_zeros(batch_size, future_len, num_vars)
        y_mark = self._prepare_time_mark(future_reference, y_mark, start=self.seq_len)
        if y_mask is None:
            y_mask = torch.isfinite(y) if y is not None else torch.ones_like(future_reference, dtype=torch.bool)
        else:
            y_mask = self._prepare_mask(future_reference, y_mask)

        x, means, stdev = self._normalize_history(x, x_mask)
        history_time, future_time = self._normalize_times(x_mark, y_mark)
        enc_input = torch.cat([x * x_mask.to(dtype=x.dtype), x_mask.to(dtype=x.dtype)], dim=2)
        encoded = self.enc(enc_input, history_time)
        qz0_mean = encoded[:, :, :self.latent_dim]
        qz0_logvar = encoded[:, :, self.latent_dim:]
        if self.kl_weight > 0.0:
            self._regularization_loss = self.kl_weight * (-0.5 * (1 + qz0_logvar - qz0_mean.pow(2) - qz0_logvar.exp())).mean()
        else:
            self._regularization_loss = None

        latent = self._sample_latent(qz0_mean, qz0_logvar)
        pred = self._decode_and_average(latent, future_time, batch_size)
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

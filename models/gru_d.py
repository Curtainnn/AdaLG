import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


MODEL_NAME = 'gru_d'
MODEL_ALIASES = ('GRU_D', 'GRUD', 'GRU-D', 'grud')
SUPPORTS_MASK_ARGS = True


class TemporalDecay(nn.Module):
    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = bool(diag)
        self.W = Parameter(torch.empty(output_size, input_size))
        self.b = Parameter(torch.empty(output_size))
        if self.diag:
            if input_size != output_size:
                raise ValueError('TemporalDecay diag=True requires input_size == output_size.')
            self.register_buffer('diag_mask', torch.eye(input_size, input_size), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta):
        weight = self.W * self.diag_mask if self.diag else self.W
        gamma = F.relu(F.linear(delta, weight, self.b))
        return torch.exp(-gamma)


class BackboneGRUD(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.rnn_cell = nn.GRUCell(self.n_features * 2 + self.hidden_size, self.hidden_size)
        self.temp_decay_h = TemporalDecay(self.n_features, self.hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(self.n_features, self.n_features, diag=True)

    def forward(
        self,
        values,
        observed_mask,
        deltas,
        empirical_mean,
        locf_values,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_steps, _ = values.shape
        hidden_state = values.new_zeros(batch_size, self.hidden_size)
        representations = []

        mask_float = observed_mask.to(dtype=values.dtype)
        for step in range(n_steps):
            x_t = values[:, step, :]
            m_t = mask_float[:, step, :]
            d_t = deltas[:, step, :]
            locf_t = locf_values[:, step, :]

            hidden_state = hidden_state * self.temp_decay_h(d_t)
            gamma_x = self.temp_decay_x(d_t)
            x_decay = gamma_x * locf_t + (1.0 - gamma_x) * empirical_mean
            x_replaced = m_t * x_t + (1.0 - m_t) * x_decay

            rnn_input = torch.cat([x_replaced, hidden_state, m_t], dim=1)
            hidden_state = self.rnn_cell(rnn_input, hidden_state)
            representations.append(hidden_state)

        return torch.stack(representations, dim=1), hidden_state


class Model(nn.Module):
    supports_mask_args = True

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(
            max(
                getattr(configs, 'pred_len', 0),
                getattr(configs, 'output_token_len', 0),
                getattr(configs, 'test_pred_len', 0),
            )
        )
        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'n_vars', 1)))
        self.features = str(getattr(configs, 'features', 'M'))
        self.hidden_size = int(getattr(configs, 'grud_hidden_size', 100))
        if self.seq_len <= 0 or self.pred_len <= 0 or self.enc_in <= 0 or self.hidden_size <= 0:
            raise ValueError('GRU-D requires positive seq_len, pred_len, enc_in, and grud_hidden_size.')

        self.backbone = BackboneGRUD(self.enc_in, self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.enc_in)

    def _normalize_time_mark(self, mark, batch_size, length, device, dtype, start=0.0):
        if mark is None:
            mark = torch.arange(length, dtype=dtype, device=device).view(1, length, 1) + float(start)
            return mark.expand(batch_size, -1, -1)
        if mark.dim() == 2:
            mark = mark.unsqueeze(-1)
        if mark.dim() != 3 or mark.shape[1] != length:
            raise ValueError('GRU-D time mark must have shape [B, L] or [B, L, 1].')
        if mark.shape[0] == 1 and batch_size > 1:
            mark = mark.expand(batch_size, -1, -1)
        if mark.shape[0] != batch_size:
            raise ValueError('GRU-D time mark batch dimension does not match x.')
        return mark.to(device=device, dtype=dtype)

    def _normalize_mask(self, values, mask):
        if mask is None:
            return torch.isfinite(values)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, values.shape[-1])
        elif mask.shape != values.shape:
            raise ValueError('GRU-D mask must match values or broadcast across variable dimension.')
        return mask.to(dtype=torch.bool)

    def _build_future_mark(self, y_mark, y, batch_size, device, dtype):
        if y_mark is not None:
            length = int(y_mark.shape[1])
            return self._normalize_time_mark(y_mark, batch_size, length, device, dtype)
        length = int(y.shape[1]) if y is not None else self.pred_len
        future_time = torch.arange(1, length + 1, dtype=dtype, device=device).view(1, length, 1)
        return future_time.expand(batch_size, -1, -1) + float(self.seq_len)

    def _compute_deltas(self, time_mark, observed_mask):
        batch_size, n_steps, _ = time_mark.shape
        num_vars = observed_mask.shape[-1]
        time = time_mark[:, :, 0]
        step_delta = time.new_zeros(batch_size, n_steps)
        if n_steps > 1:
            step_delta[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        deltas = time.new_zeros(batch_size, n_steps, num_vars)
        for step in range(1, n_steps):
            previous_observed = observed_mask[:, step - 1, :]
            increment = step_delta[:, step].unsqueeze(-1)
            deltas[:, step, :] = torch.where(
                previous_observed,
                increment,
                deltas[:, step - 1, :] + increment,
            )
        return deltas

    def _empirical_mean(self, values, observed_mask):
        weight = observed_mask.to(dtype=values.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        return (values * weight).sum(dim=1) / count

    def _locf(self, values, observed_mask, empirical_mean):
        batch_size, n_steps, num_vars = values.shape
        last_value = empirical_mean
        locf_values = []
        for step in range(n_steps):
            locf_values.append(last_value)
            last_value = torch.where(observed_mask[:, step, :], values[:, step, :], last_value)
        return torch.stack(locf_values, dim=1).reshape(batch_size, n_steps, num_vars)

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None, y=None, **kwargs):
        del kwargs
        batch_size, history_len, num_vars = x.shape
        if num_vars != self.enc_in:
            raise ValueError(f'GRU-D expected {self.enc_in} variables, got {num_vars}.')

        x_mask = self._normalize_mask(x, x_mask)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.where(x_mask, x, torch.zeros_like(x))
        x_mark = self._normalize_time_mark(x_mark, batch_size, history_len, x.device, x.dtype)

        future_mark = self._build_future_mark(y_mark, y, batch_size, x.device, x.dtype)
        future_len = int(future_mark.shape[1])
        future_values = x.new_zeros(batch_size, future_len, num_vars)
        future_mask = torch.zeros(batch_size, future_len, num_vars, dtype=torch.bool, device=x.device)

        values = torch.cat([x, future_values], dim=1)
        observed_mask = torch.cat([x_mask, future_mask], dim=1)
        time_mark = torch.cat([x_mark, future_mark], dim=1)

        deltas = self._compute_deltas(time_mark, observed_mask)
        empirical_mean = self._empirical_mean(x, x_mask)
        locf_values = self._locf(values, observed_mask, empirical_mean)
        representations, _ = self.backbone(values, observed_mask, deltas, empirical_mean, locf_values)
        pred = self.output_projection(representations[:, -future_len:, :])

        if y is None:
            return pred

        if y_mask is None:
            y_mask = torch.isfinite(y)
        else:
            y_mask = y_mask.to(dtype=torch.bool)

        if self.features == 'MS':
            pred = pred[:, :, -1:]
            y = y[:, :, -1:]
            y_mask = y_mask[:, :, -1:]

        return {
            'pred': pred,
            'true': y,
            'mask': y_mask,
        }

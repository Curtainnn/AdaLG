import torch
import torch.nn as nn


MODEL_NAME = 'dlinear'
MODEL_ALIASES = ('DLinear', 'D-Linear')
SUPPORTS_MASK_ARGS = True


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 1:
            raise ValueError('DLinear moving average kernel size must be positive.')
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        if pad > 0:
            front = x[:, 0:1, :].repeat(1, pad, 1)
            end = x[:, -1:, :].repeat(1, pad, 1)
            x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


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
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError('DLinear requires positive seq_len and pred_len/output_token_len.')

        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'n_vars', 1)))
        if self.enc_in <= 0:
            raise ValueError('DLinear requires a positive enc_in/n_vars.')

        self.features = str(getattr(configs, 'features', 'M'))
        self.individual = bool(getattr(configs, 'dlinear_individual', False))
        self.decomposition = SeriesDecomp(int(getattr(configs, 'dlinear_moving_avg', 25)))

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def _normalize_mask(self, x, x_mask):
        if x_mask is None:
            return torch.isfinite(x)
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        elif x_mask.dim() == 3 and x_mask.shape[-1] == 1:
            x_mask = x_mask.expand(-1, -1, x.shape[-1])
        elif x_mask.shape != x.shape:
            raise ValueError('DLinear x_mask must match x or broadcast across variable dimension.')
        return x_mask.to(dtype=torch.bool)

    def _fit_input_length(self, x, x_mask):
        cur_len = int(x.shape[1])
        if cur_len == self.seq_len:
            return x, x_mask
        if cur_len > self.seq_len:
            return x[:, -self.seq_len:, :], x_mask[:, -self.seq_len:, :]

        pad_len = self.seq_len - cur_len
        x_pad = x.new_zeros(x.shape[0], pad_len, x.shape[2])
        mask_pad = torch.zeros(x_mask.shape[0], pad_len, x_mask.shape[2], dtype=torch.bool, device=x_mask.device)
        return torch.cat([x_pad, x], dim=1), torch.cat([mask_pad, x_mask], dim=1)

    def _prepare_input(self, x, x_mask):
        x_mask = self._normalize_mask(x, x_mask)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.where(x_mask, x, torch.zeros_like(x))
        return self._fit_input_length(x, x_mask)

    def encoder(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            batch_size, num_vars, _ = seasonal_init.shape
            if num_vars != self.enc_in:
                raise ValueError(f'DLinear expected {self.enc_in} variables, got {num_vars}.')
            seasonal_output = seasonal_init.new_zeros(batch_size, num_vars, self.pred_len)
            trend_output = trend_init.new_zeros(batch_size, num_vars, self.pred_len)
            for var_id in range(num_vars):
                seasonal_output[:, var_id, :] = self.linear_seasonal[var_id](seasonal_init[:, var_id, :])
                trend_output[:, var_id, :] = self.linear_trend[var_id](trend_init[:, var_id, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        return (seasonal_output + trend_output).permute(0, 2, 1)

    def _align_prediction_to_targets(self, pred, y_mark, target_len):
        if y_mark is None:
            if target_len > pred.shape[1]:
                raise ValueError(f'DLinear produced {pred.shape[1]} steps but target requires {target_len}.')
            return pred[:, :target_len, :]

        if y_mark.dim() == 3:
            target_time = y_mark[:, :, 0]
        elif y_mark.dim() == 2:
            target_time = y_mark
        else:
            raise ValueError('DLinear y_mark must have shape [B, T] or [B, T, 1].')

        if int(target_time.shape[1]) != int(target_len):
            raise ValueError(
                f'DLinear target_len={target_len} does not match y_mark length={target_time.shape[1]}.'
            )

        # Map continuous future query times to fixed DLinear horizon slots.
        # The first forecast slot represents (seq_len, seq_len + 1].
        horizon_slot = torch.ceil(target_time - float(self.seq_len)).long() - 1
        horizon_slot = horizon_slot.clamp(min=0, max=pred.shape[1] - 1)
        gather_index = horizon_slot.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
        return pred.gather(dim=1, index=gather_index)

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None, y=None, **kwargs):
        del x_mark, kwargs
        x, _ = self._prepare_input(x, x_mask)
        pred = self.encoder(x)

        target_len = None
        if y is not None:
            target_len = int(y.shape[1])
        elif y_mark is not None:
            target_len = int(y_mark.shape[1])

        if target_len is not None:
            pred = self._align_prediction_to_targets(pred, y_mark, target_len)

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

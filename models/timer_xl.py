import torch
from torch import nn

from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.timer_xl_point_frontend import TimerXLPointFrontend
from layers.timer_xl_query_head import TimerXLQueryPointHead


MODEL_NAME = 'timer_xl'
MODEL_ALIASES = ('Timer-XL', 'timer-xl', 'TimerXL')
ACCEPTS_BATCH_DICT = True
SUPPORTS_MASK_ARGS = True
ADAPTATION_BACKBONE_PREFIXES = ('blocks.',)


class Model(nn.Module):
    """Timer-XL-style long-context Transformer baseline."""
    accepts_batch_dict = True
    supports_mask_args = True
    use_point_representation = True

    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        num_vars = int(getattr(configs, 'point_num_vars', 0) or getattr(configs, 'n_vars', 0))
        if num_vars <= 0:
            raise ValueError('Timer-XL point mode requires a positive point_num_vars or n_vars.')
        point_num_nodes = (
            getattr(configs, 'node_num', None)
            if getattr(configs, 'point_use_node_embedding', False)
            else None
        )
        self.point_frontend = TimerXLPointFrontend(
            input_token_len=self.input_token_len,
            hidden_dim=configs.point_hidden_dim,
            token_subbins=getattr(configs, 'point_token_subbins', 4),
            time_dim=configs.point_time_dim,
            num_vars=num_vars,
            num_nodes=point_num_nodes,
            causal_ode_state_dim=getattr(configs, "causal_ode_state_dim", 32),
            causal_ode_hidden=getattr(configs, "causal_ode_hidden", 128),
            causal_ode_constraint_strength=getattr(configs, "causal_ode_constraint_strength", 0.0),
            causal_ode_h_weight=getattr(configs, "causal_ode_h_weight", 1e-3),
            causal_ode_l1_weight=getattr(configs, "causal_ode_l1_weight", 1e-4),
            causal_ode_diag_weight=getattr(configs, "causal_ode_diag_weight", 1e-3),
            causal_ode_laplacian_weight=getattr(configs, "causal_ode_laplacian_weight", 1e-4),
            causal_ode_adjacency_scale=getattr(configs, "causal_ode_adjacency_scale", 1.0),
            causal_ode_gate_init=getattr(configs, "causal_ode_gate_init", -4.0),
            causal_ode_steps=getattr(configs, "causal_ode_steps", 8),
            causal_prior_window=getattr(configs, "causal_prior_window", 30),
        )
        self.point_proj = nn.Linear(configs.point_hidden_dim, configs.d_model)
        self.causal_context_proj = nn.Linear(configs.point_hidden_dim, configs.d_model)
        self.point_causal_context_mode = str(getattr(configs, 'point_causal_context_mode', 'normal'))
        if self.point_causal_context_mode not in {'normal', 'zero', 'shuffled'}:
            raise ValueError(
                f"Unsupported point_causal_context_mode={self.point_causal_context_mode!r}; "
                "expected 'normal', 'zero', or 'shuffled'."
            )
        self.query_head = TimerXLQueryPointHead(
            d_model=configs.d_model,
            input_token_len=self.input_token_len,
            time_dim=configs.point_time_dim,
            num_vars=num_vars,
            dropout=configs.dropout,
            num_nodes=point_num_nodes,
            local_fusion_mode=getattr(configs, 'point_local_fusion_mode', 'dual_gate'),
            local_fusion_k=getattr(configs, 'point_local_fusion_k', getattr(configs, 'point_knn', 8)),
            local_gate_bias=getattr(configs, 'point_local_gate_bias', -4.0),
            dual_gate_bias=getattr(configs, 'point_dual_gate_bias', -1.0),
            point_gate_max=getattr(configs, 'point_gate_max', 1.0),
            causal_context_mode=self.point_causal_context_mode,
            causal_local_alpha=getattr(configs, 'causal_local_alpha', 0.0),
            causal_local_beta=getattr(configs, 'causal_local_beta', 1.0),
            causal_local_expand_pool=getattr(configs, 'causal_local_expand_pool', False),
            causal_prior_mode=getattr(configs, 'causal_prior_mode', 'normal'),
            force_fusion=getattr(configs, 'point_force_fusion', 'auto'),
            fixed_gate=getattr(configs, 'point_fixed_gate', None),
        )
        self.output_attention = configs.output_attention
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            True,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                            d_model=configs.d_model,
                            num_heads=configs.n_heads,
                            covariate=configs.covariate,
                            flash_attention=configs.flash_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.use_norm = configs.use_norm


    def get_regularization_loss(self):
        return self.point_frontend.get_regularization_loss()

    def get_causal_ode_gate(self):
        if hasattr(self.point_frontend, 'get_causal_ode_gate'):
            return self.point_frontend.get_causal_ode_gate()
        return None

    def _project_causal_context(self, causal_context_raw):
        causal_context = self.causal_context_proj(causal_context_raw)
        if not hasattr(self, "_debug_causal_context_printed"):
            print(
                "[DEBUG] causal_context_raw_shape={} causal_context_shape={} "
                "mode={} raw_mean={:.6f} projected_mean={:.6f}".format(
                    tuple(causal_context_raw.shape),
                    tuple(causal_context.shape),
                    self.point_causal_context_mode,
                    causal_context_raw.detach().mean().cpu().item() if causal_context_raw.numel() > 0 else 0.0,
                    causal_context.detach().mean().cpu().item() if causal_context.numel() > 0 else 0.0,
                ),
                flush=True,
            )
            self._debug_causal_context_printed = True
        return causal_context

    def _normalize_dense_input(self, x, x_mask=None):
        valid_mask = x_mask.to(dtype=torch.bool) if x_mask is not None else torch.isfinite(x)
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(safe_x.dtype)
        means = (safe_x * valid_mask.to(safe_x.dtype)).sum(dim=1, keepdim=True) / valid_count
        centered_x = torch.where(valid_mask, safe_x - means, torch.zeros_like(safe_x))
        variance = centered_x.pow(2).sum(dim=1, keepdim=True) / valid_count
        stdev = torch.sqrt(variance + 1e-5)
        normalized_x = centered_x / stdev
        return normalized_x, means.detach(), stdev

    def _normalize_point_batch(self, batch):
        batch_size = int(batch['batch_size'])
        num_vars = int(batch['num_vars'])
        obs_value = batch['obs_value']
        obs_batch_id = batch['obs_batch_id'].long()
        obs_var_id = batch['obs_var_id'].long()

        group_idx = obs_batch_id * num_vars + obs_var_id
        group_size = batch_size * num_vars
        group_sum = obs_value.new_zeros(group_size)
        group_sum_sq = obs_value.new_zeros(group_size)
        group_count = obs_value.new_zeros(group_size)

        group_sum.index_add_(0, group_idx, obs_value)
        group_sum_sq.index_add_(0, group_idx, obs_value * obs_value)
        group_count.index_add_(0, group_idx, obs_value.new_ones(obs_value.shape[0]))

        safe_count = group_count.clamp_min(1.0)
        means = (group_sum / safe_count).view(batch_size, num_vars)
        variance = (group_sum_sq / safe_count) - means.reshape(-1).pow(2)
        stdev = torch.sqrt(variance.clamp_min(0.0).view(batch_size, num_vars) + 1e-5)

        normalized_batch = dict(batch)
        normalized_batch['obs_value'] = (obs_value - means.view(-1)[group_idx]) / stdev.view(-1)[group_idx]
        return normalized_batch, means.detach(), stdev

    def _build_input_tokens(self, x, x_mark=None, x_mask=None, seq_len=None):
        if isinstance(x, dict):
            token_feature, token_mask, n_tokens, causal_context_raw, causal_prior = self.point_frontend(x)
            embed_out = self.point_proj(token_feature)
            causal_context = self._project_causal_context(causal_context_raw)
            return embed_out, token_mask, int(x['batch_size']), int(x['num_vars']), n_tokens, causal_context, causal_prior

        batch_size, _, num_vars = x.shape
        token_feature, token_mask, n_tokens, causal_context_raw, causal_prior = self.point_frontend(
            x,
            x_mark=x_mark,
            x_mask=x_mask,
            seq_len=seq_len,
        )
        embed_out = self.point_proj(token_feature)
        causal_context = self._project_causal_context(causal_context_raw)
        return embed_out, token_mask, batch_size, num_vars, n_tokens, causal_context, causal_prior

    def _run_backbone(self, embed_out, batch_size, num_vars, n_tokens):
        embed_out = embed_out.reshape(batch_size, num_vars * n_tokens, -1)
        embed_out, attns = self.blocks(embed_out, n_vars=num_vars, n_tokens=n_tokens)
        token_feature = embed_out.reshape(batch_size, num_vars, n_tokens, -1)
        return token_feature, attns

    def _predict_dense_queries(
        self,
        token_feature,
        token_mask,
        batch,
        seq_len,
        history_x,
        history_x_mark,
        causal_context,
        causal_prior,
    ):
        target = batch['y']
        target_mask = batch['y_mask'].to(dtype=torch.bool)
        batch_size, pred_len, num_vars = target.shape
        pred_dense = token_feature.new_zeros(batch_size, pred_len, num_vars)

        valid_query_idx = target_mask.nonzero(as_tuple=False)
        if valid_query_idx.numel() == 0:
            return pred_dense

        query_batch_id = valid_query_idx[:, 0].long()
        query_row_id = valid_query_idx[:, 1].long()
        query_var_id = valid_query_idx[:, 2].long()
        query_time = batch['y_mark'][query_batch_id, query_row_id, 0].to(token_feature.dtype)
        history_obs_idx = batch['x_mask'].to(dtype=torch.bool).nonzero(as_tuple=False)
        history_batch_id = history_obs_idx[:, 0].long()
        history_time = history_x_mark[history_batch_id, history_obs_idx[:, 1], 0].to(token_feature.dtype)
        history_var_id = history_obs_idx[:, 2].long()
        history_value = history_x[history_batch_id, history_obs_idx[:, 1], history_var_id]

        pred_value = self.query_head(
            token_feature=token_feature,
            token_mask=token_mask,
            query_time=query_time,
            query_var_id=query_var_id,
            query_batch_id=query_batch_id,
            seq_len=seq_len,
            history_time=history_time,
            history_var_id=history_var_id,
            history_batch_id=history_batch_id,
            history_value=history_value,
            causal_context=causal_context,
            causal_prior=causal_prior,
        )
        pred_dense[query_batch_id, query_row_id, query_var_id] = pred_value
        return pred_dense

    def forecast_dense_batch(self, batch):
        x = batch['x']
        x_mark = batch['x_mark']
        x_mask = batch['x_mask']
        seq_len = int(batch.get('obs_seq_len', x.shape[1]))
        if x_mark is None:
            x_mark = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1).expand(x.shape[0], -1, -1)
        elif x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(-1)

        if self.use_norm:
            x, means, stdev = self._normalize_dense_input(x, x_mask=x_mask)
        else:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            means = None
            stdev = None

        embed_out, token_mask, batch_size, num_vars, n_tokens, causal_context, causal_prior = self._build_input_tokens(
            x,
            x_mark=x_mark,
            x_mask=x_mask,
            seq_len=seq_len,
        )
        token_feature, attns = self._run_backbone(embed_out, batch_size, num_vars, n_tokens)
        pred_dense = self._predict_dense_queries(
            token_feature,
            token_mask,
            batch,
            seq_len=seq_len,
            history_x=x,
            history_x_mark=x_mark,
            causal_context=causal_context,
            causal_prior=causal_prior,
        )

        if self.use_norm:
            pred_dense = pred_dense * stdev + means
        if self.output_attention:
            return pred_dense, attns
        return pred_dense

    def forecast(self, x, x_mark, y_mark, x_mask=None, y_mask=None):
        batch_size, _, num_vars = x.shape
        if y_mark.dim() == 2:
            y_mark = y_mark.unsqueeze(-1)
        pred_len = y_mark.shape[1]
        if x_mask is None:
            x_mask = torch.isfinite(x)
        if y_mask is None:
            y_mask = torch.ones(batch_size, pred_len, num_vars, dtype=torch.bool, device=x.device)
        batch = {
            'x': x,
            'x_mark': x_mark,
            'x_mask': x_mask,
            'y': x.new_zeros(batch_size, pred_len, num_vars),
            'y_mark': y_mark,
            'y_mask': y_mask,
            'obs_seq_len': x.shape[1],
            'num_vars': num_vars,
        }
        return self.forecast_dense_batch(batch)

    def forecast_point_batch(self, batch):
        if self.use_norm:
            batch, means, stdev = self._normalize_point_batch(batch)

        embed_out, token_mask, batch_size, num_vars, n_tokens, causal_context, causal_prior = self._build_input_tokens(batch)
        token_feature, attns = self._run_backbone(embed_out, batch_size, num_vars, n_tokens)
        pred_value = self.query_head(
            token_feature=token_feature,
            token_mask=token_mask,
            query_time=batch['query_time'],
            query_var_id=batch['query_var_id'].long(),
            query_batch_id=batch['query_batch_id'].long(),
            seq_len=int(batch['obs_seq_len']),
            query_node_id=batch.get('query_node_id'),
            history_time=batch['obs_time'],
            history_var_id=batch['obs_var_id'].long(),
            history_batch_id=batch['obs_batch_id'].long(),
            history_value=batch['obs_value'],
            history_node_id=batch.get('obs_node_id'),
            causal_context=causal_context,
            causal_prior=causal_prior,
        )

        if self.use_norm:
            query_batch_id = batch['query_batch_id'].long()
            query_var_id = batch['query_var_id'].long()
            pred_value = pred_value * stdev[query_batch_id, query_var_id] + means[query_batch_id, query_var_id]

        if self.output_attention:
            return pred_value, attns
        return pred_value

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None):
        if isinstance(x, dict):
            if 'x' in x:
                return self.forecast_dense_batch(x)
            return self.forecast_point_batch(x)
        return self.forecast(x, x_mark, y_mark, x_mask=x_mask, y_mask=y_mask)

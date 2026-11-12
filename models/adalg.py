import torch
from torch import nn

from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.mechanism_encoder import ObservationPointEncoder
from layers.point_to_token import PointToTokenAggregator
from layers.adalg_query_point_head import QueryPointHead
from layers.causal_ode import CausalODEFeaturePropagation
from utils.local_reference_cache import LocalReferenceCache


MODEL_NAME = "AdaLG"
MODEL_ALIASES = ("adalg", "ada_lg", "ada-lg")
ACCEPTS_BATCH_DICT = True
SUPPORTS_MASK_ARGS = True
ADAPTATION_BACKBONE_PREFIXES = ("blocks.",)


class AdaLGPointFrontend(nn.Module):
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
            dropout=0.0,
            ode_steps=int(causal_ode_steps),
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
        inferred_seq_len = (
            int(seq_len)
            if seq_len is not None
            else (int(torch.ceil(x_mark[..., 0].max()).item()) + 1 if x_mark.numel() > 0 else x.shape[1])
        )
        return self._encode_points(
            obs_value=obs_value,
            obs_time=obs_time,
            obs_var_id=obs_var_id,
            obs_batch_id=obs_batch_id,
            batch_size=batch_size,
            num_vars=num_vars,
            seq_len=inferred_seq_len,
        )


class Model(nn.Module):
    history_boundary_policy = "closed_history_v2"
    accepts_batch_dict = True
    supports_mask_args = True
    use_point_representation = True

    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        num_vars = int(getattr(configs, "point_num_vars", 0) or getattr(configs, "n_vars", 0))
        if num_vars <= 0:
            raise ValueError("AdaLG requires a positive point_num_vars or n_vars.")
        point_num_nodes = (
            getattr(configs, "node_num", None)
            if getattr(configs, "point_use_node_embedding", False)
            else None
        )
        self.point_frontend = AdaLGPointFrontend(
            input_token_len=self.input_token_len,
            hidden_dim=configs.point_hidden_dim,
            token_subbins=getattr(configs, "point_token_subbins", 4),
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
        print(f"[ARG] causal_ode_gate_init={getattr(configs, 'causal_ode_gate_init', -4.0)}  causal_ode_steps={getattr(configs, 'causal_ode_steps', 8)}", flush=True)
        self.point_disable_local_branch = bool(getattr(configs, "point_disable_local_branch", False))
        self.point_gate_mode = str(getattr(configs, "point_gate_mode", "utility"))
        self.point_use_causal_gate_context = not bool(
            getattr(configs, "point_disable_causal_gate_context", False)
        )
        self.eval_input_observation_drop_rate = 0.0
        self.eval_input_observation_drop_seed = 0
        self._eval_input_observation_drop_call_count = 0
        self.point_proj = nn.Linear(configs.point_hidden_dim, configs.d_model)
        self.causal_context_proj = (
            None
            if self.point_disable_local_branch
            else nn.Linear(configs.point_hidden_dim, configs.d_model)
        )
        self.point_causal_context_mode = str(getattr(configs, "point_causal_context_mode", "normal"))
        if self.point_causal_context_mode not in {"normal", "zero", "shuffled"}:
            raise ValueError(
                f"Unsupported point_causal_context_mode={self.point_causal_context_mode!r}; "
                "expected 'normal', 'zero', or 'shuffled'."
            )
        head_class = QueryPointHead
        extra_head_args = {}
        if getattr(configs, 'point_prediction_variant', 'adalg') == 'raw_recent_concat':
            from layers.raw_recent_concat import RawRecentConcatHead
            head_class = RawRecentConcatHead
            extra_head_args['concat_hidden'] = getattr(configs, 'raw_concat_hidden', 0)
        self.query_head = head_class(
            d_model=configs.d_model,
            input_token_len=self.input_token_len,
            time_dim=configs.point_time_dim,
            num_vars=num_vars,
            dropout=configs.dropout,
            num_nodes=point_num_nodes,
            local_fusion_mode=getattr(configs, "point_local_fusion_mode", "dual_gate"),
            local_fusion_k=getattr(configs, "point_local_fusion_k", getattr(configs, "point_knn", 8)),
            local_gate_bias=getattr(configs, "point_local_gate_bias", -4.0),
            dual_gate_bias=getattr(configs, "point_dual_gate_bias", -1.0),
            point_gate_max=getattr(configs, "point_gate_max", 1.0),
            causal_context_mode=self.point_causal_context_mode,
            causal_local_alpha=getattr(configs, "causal_local_alpha", 0.0),
            causal_local_beta=getattr(configs, "causal_local_beta", 1.0),
            causal_local_expand_pool=getattr(configs, "causal_local_expand_pool", False),
            causal_prior_mode=getattr(configs, "causal_prior_mode", "normal"),
            force_fusion=getattr(configs, "point_force_fusion", "auto"),
            fixed_gate=getattr(configs, "point_fixed_gate", None),
            disable_local_branch=self.point_disable_local_branch,
            gate_mode=self.point_gate_mode,
            use_causal_gate_context=self.point_use_causal_gate_context,
            gate_prediction_features=bool(getattr(configs, "point_gate_prediction_features", False)),
            separate_utility_head=bool(getattr(configs, "point_separate_utility_head", False)),
            local_branch_source=getattr(configs, "point_local_branch_source", "raw"),
            local_encoder=getattr(configs, "point_local_encoder", "summary"),
            gate_availability_features=not bool(getattr(configs, "point_disable_gate_availability_features", False)),
            recent_strict_past=bool(getattr(configs, 'point_recent_strict_past', False)),
            recent_analysis=bool(getattr(configs, 'raw_recent_query_export', False)),
            **extra_head_args,
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
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.use_norm = configs.use_norm
        self._eval_local_reference = LocalReferenceCache()

    def begin_eval_local_reference(self):
        if self.training:
            raise RuntimeError('Local reference caching is only available in eval mode.')
        self._eval_local_reference.begin()
        self.query_head._eval_local_reference.begin()

    def rewind_eval_local_reference(self):
        self._eval_local_reference.rewind()
        self.query_head._eval_local_reference.rewind()

    def end_eval_local_reference(self):
        self._eval_local_reference.end()
        self.query_head._eval_local_reference.end()

    def _lookup_eval_local_reference(self, batch):
        if self._eval_local_reference.active and self.training:
            raise RuntimeError('Cannot train while a local evaluation reference is active.')
        reference = batch['x'] if 'x' in batch else batch['obs_value']
        # Include values and target/query masks: replay must not conceal a real
        # input change or accidentally pair different samples.
        return self._eval_local_reference.lookup(batch, reference.device)


    def get_regularization_loss(self):
        return self.point_frontend.get_regularization_loss()

    def get_causal_ode_gate(self):
        if hasattr(self.point_frontend, "get_causal_ode_gate"):
            return self.point_frontend.get_causal_ode_gate()
        return None

    def set_eval_input_observation_drop(self, drop_rate=0.0, seed=0):
        drop_rate = float(drop_rate)
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError("eval input observation drop rate must be in [0, 1].")
        self.eval_input_observation_drop_rate = drop_rate
        self.eval_input_observation_drop_seed = int(seed)
        self._eval_input_observation_drop_call_count = 0

    def _apply_eval_input_observation_drop(self, batch):
        rate = float(self.eval_input_observation_drop_rate)
        if self.training or rate <= 0.0:
            return batch
        count = int(batch["x_mask"].numel()) if "x" in batch else int(batch["obs_value"].numel())
        seed = int(self.eval_input_observation_drop_seed) + int(self._eval_input_observation_drop_call_count)
        self._eval_input_observation_drop_call_count += 1
        reference = batch.get("x_mask") if "x" in batch else batch["obs_value"]
        if reference is None:
            reference = batch["x"]
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(seed)
        keep = torch.rand(count, device=reference.device, generator=generator) >= rate
        dropped = dict(batch)
        if "x" in batch:
            original_mask = batch.get("x_mask")
            if original_mask is None:
                original_mask = torch.isfinite(batch["x"])
            dropped["x_mask"] = original_mask.to(dtype=torch.bool) & keep.reshape(original_mask.shape)
        else:
            obs_count = int(batch["obs_value"].numel())
            keep = keep[:obs_count]
            for key, value in batch.items():
                if key.startswith("obs_") and torch.is_tensor(value) and value.shape[:1] == (obs_count,):
                    dropped[key] = value[keep]
        return dropped

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
        batch_size = int(batch["batch_size"])
        num_vars = int(batch["num_vars"])
        obs_value = batch["obs_value"]
        obs_batch_id = batch["obs_batch_id"].long()
        obs_var_id = batch["obs_var_id"].long()

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
        normalized_batch["obs_value"] = (obs_value - means.view(-1)[group_idx]) / stdev.view(-1)[group_idx]
        return normalized_batch, means.detach(), stdev

    def _project_causal_context(self, causal_context_raw):
        if self.causal_context_proj is None:
            if not hasattr(self, "_debug_causal_context_printed"):
                print("[DEBUG] point_disable_local_branch=True; causal_context_proj disabled", flush=True)
                self._debug_causal_context_printed = True
            return None

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

    def _build_input_tokens(self, x, x_mark=None, x_mask=None, seq_len=None):
        if isinstance(x, dict):
            token_feature, token_mask, n_tokens, causal_context_raw, causal_prior = self.point_frontend(x)
            embed_out = self.point_proj(token_feature)
            causal_context = self._project_causal_context(causal_context_raw)
            return embed_out, token_mask, int(x["batch_size"]), int(x["num_vars"]), n_tokens, causal_context, causal_prior

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
        target = batch["y"]
        target_mask = batch["y_mask"].to(dtype=torch.bool)
        batch_size, pred_len, num_vars = target.shape
        pred_dense = token_feature.new_zeros(batch_size, pred_len, num_vars)
        aux_dense = {
            "global_pred": token_feature.new_zeros(batch_size, pred_len, num_vars),
            "local_pred": token_feature.new_zeros(batch_size, pred_len, num_vars),
            "gate": token_feature.new_zeros(batch_size, pred_len, num_vars),
            "gate_logit": token_feature.new_zeros(batch_size, pred_len, num_vars),
            "utility_logit": token_feature.new_zeros(batch_size, pred_len, num_vars),
            "has_local_history": torch.zeros(batch_size, pred_len, num_vars, dtype=torch.bool, device=token_feature.device),
        }

        valid_query_idx = target_mask.nonzero(as_tuple=False)
        if valid_query_idx.numel() == 0:
            return pred_dense, aux_dense

        query_batch_id = valid_query_idx[:, 0].long()
        query_row_id = valid_query_idx[:, 1].long()
        query_var_id = valid_query_idx[:, 2].long()
        query_time = batch["y_mark"][query_batch_id, query_row_id, 0].to(token_feature.dtype)
        history_obs_idx = batch["x_mask"].to(dtype=torch.bool).nonzero(as_tuple=False)
        history_batch_id = history_obs_idx[:, 0].long()
        history_time = history_x_mark[history_batch_id, history_obs_idx[:, 1], 0].to(token_feature.dtype)
        history_var_id = history_obs_idx[:, 2].long()
        history_value = history_x[history_batch_id, history_obs_idx[:, 1], history_var_id]

        pred_value, query_aux = self.query_head(
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
            return_aux=True,
        )
        pred_dense[query_batch_id, query_row_id, query_var_id] = pred_value
        for key in query_aux:
            if key == 'gate_stats':
                continue
            if key not in aux_dense:
                aux_dense[key] = pred_dense.new_zeros(pred_dense.shape)
            aux_dense[key][query_batch_id, query_row_id, query_var_id] = query_aux[key]
        return pred_dense, aux_dense

    def forecast_dense_batch(self, batch):
        batch = self._apply_eval_input_observation_drop(batch)
        ticket, reference = self._lookup_eval_local_reference(batch)
        x = batch["x"]
        x_mark = batch["x_mark"]
        x_mask = batch["x_mask"]
        seq_len = int(batch.get("obs_seq_len", x.shape[1]))
        if x_mark is None:
            x_mark = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1).expand(x.shape[0], -1, -1)
        elif x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(-1)

        if reference is not None:
            x, means, stdev, token_feature, token_mask, causal_context, causal_prior, attns = reference
        elif self.use_norm:
            x, means, stdev = self._normalize_dense_input(x, x_mask=x_mask)
        else:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            means = None
            stdev = None

        if reference is None:
            embed_out, token_mask, batch_size, num_vars, n_tokens, causal_context, causal_prior = self._build_input_tokens(
                x, x_mark=x_mark, x_mask=x_mask, seq_len=seq_len,
            )
            token_feature, attns = self._run_backbone(embed_out, batch_size, num_vars, n_tokens)
            self._eval_local_reference.store(ticket, (x, means, stdev, token_feature, token_mask, causal_context, causal_prior, attns))
        pred_dense, aux_dense = self._predict_dense_queries(
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
            for key in ("global_pred", "local_pred"):
                aux_dense[key] = aux_dense[key] * stdev + means
        if self.output_attention:
            return {"pred": pred_dense, "aux": aux_dense, "attn": attns}
        return {"pred": pred_dense, "aux": aux_dense}

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
            "x": x,
            "x_mark": x_mark,
            "x_mask": x_mask,
            "y": x.new_zeros(batch_size, pred_len, num_vars),
            "y_mark": y_mark,
            "y_mask": y_mask,
            "obs_seq_len": x.shape[1],
            "num_vars": num_vars,
        }
        return self.forecast_dense_batch(batch)

    def forecast_point_batch(self, batch):
        batch = self._apply_eval_input_observation_drop(batch)
        ticket, reference = self._lookup_eval_local_reference(batch)
        if reference is not None:
            batch, means, stdev, token_feature, token_mask, causal_context, causal_prior, attns = reference
        else:
            means, stdev = None, None
            if self.use_norm:
                batch, means, stdev = self._normalize_point_batch(batch)
            embed_out, token_mask, batch_size, num_vars, n_tokens, causal_context, causal_prior = self._build_input_tokens(batch)
            token_feature, attns = self._run_backbone(embed_out, batch_size, num_vars, n_tokens)
            self._eval_local_reference.store(ticket, (batch, means, stdev, token_feature, token_mask, causal_context, causal_prior, attns))
        pred_value, query_aux = self.query_head(
            token_feature=token_feature,
            token_mask=token_mask,
            query_time=batch["query_time"],
            query_var_id=batch["query_var_id"].long(),
            query_batch_id=batch["query_batch_id"].long(),
            seq_len=int(batch["obs_seq_len"]),
            query_node_id=batch.get("query_node_id"),
            history_time=batch["obs_time"],
            history_var_id=batch["obs_var_id"].long(),
            history_batch_id=batch["obs_batch_id"].long(),
            history_value=batch["obs_value"],
            history_node_id=batch.get("obs_node_id"),
            causal_context=causal_context,
            causal_prior=causal_prior,
            return_aux=True,
        )

        if self.use_norm:
            query_batch_id = batch["query_batch_id"].long()
            query_var_id = batch["query_var_id"].long()
            pred_value = pred_value * stdev[query_batch_id, query_var_id] + means[query_batch_id, query_var_id]
            for key in ("global_pred", "local_pred"):
                query_aux[key] = (
                    query_aux[key] * stdev[query_batch_id, query_var_id]
                    + means[query_batch_id, query_var_id]
                )

        if self.output_attention:
            return {"pred": pred_value, "aux": query_aux, "attn": attns}
        return {"pred": pred_value, "aux": query_aux}

    def forward(self, x, x_mark=None, y_mark=None, x_mask=None, y_mask=None):
        if isinstance(x, dict):
            if "x" in x:
                return self.forecast_dense_batch(x)
            return self.forecast_point_batch(x)
        return self.forecast(x, x_mark, y_mark, x_mask=x_mask, y_mask=y_mask)


ObservationMechanismFrontend = AdaLGPointFrontend

import torch
from torch import nn

from layers.local_residual_adapter import LocalResidualAdapter
from layers.raw_local_encoder import RawLocalEncoder
from utils.local_reference_cache import LocalReferenceCache


class QueryPointHead(nn.Module):
    def __init__(
        self,
        d_model,
        input_token_len,
        time_dim,
        num_vars,
        dropout=0.1,
        num_nodes=None,
        local_fusion_mode="dual_gate",
        local_fusion_k=8,
        local_gate_bias=-4.0,
        dual_gate_bias=-1.0,
        point_gate_max=1.0,
        causal_context_mode="normal",
        causal_local_alpha=0.0,
        causal_local_beta=1.0,
        causal_local_expand_pool=False,
        causal_prior_mode="normal",
        force_fusion="auto",
        fixed_gate=None,
        disable_local_branch=False,
        eval_local_observation_drop_rate=0.0,
        eval_local_observation_drop_seed=0,
        gate_mode="utility",
        use_causal_gate_context=True,
        gate_prediction_features=False,
        separate_utility_head=False,
        local_branch_source="raw",
        local_encoder="summary",
        gate_availability_features=True,
        recent_strict_past=False,
        recent_analysis=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_token_len = input_token_len
        self.local_fusion_mode = str(local_fusion_mode)
        self.force_fusion = str(force_fusion)
        self.disable_local_branch = bool(disable_local_branch)
        self.gate_mode = str(gate_mode)
        self.use_causal_gate_context = bool(use_causal_gate_context)
        self.gate_prediction_features = bool(gate_prediction_features)
        self.separate_utility_head = bool(separate_utility_head)
        self.local_branch_source = str(local_branch_source)
        self.local_encoder = str(local_encoder)
        self.gate_availability_features = bool(gate_availability_features)
        self.recent_strict_past = bool(recent_strict_past)
        self.recent_analysis = bool(recent_analysis)
        self.recent_k = int(local_fusion_k)
        if self.local_branch_source not in {"raw", "shared_encoder"}:
            raise ValueError("local_branch_source must be raw or shared_encoder.")
        if self.local_encoder not in {"summary", "attention"}:
            raise ValueError("local_encoder must be summary or attention.")
        if self.local_branch_source == "raw" and self.local_encoder == "attention" and causal_local_expand_pool:
            raise ValueError("The independent raw local encoder uses same-variable observations only.")
        self.eval_local_degradation_mode = "drop"
        self._eval_local_reference = LocalReferenceCache()
        if self.gate_mode not in {"ordinary", "utility", "fixed", "global", "local"}:
            raise ValueError("gate_mode must be ordinary, utility, fixed, global, or local.")
        self.causal_context_mode = str(causal_context_mode)
        self.causal_local_alpha = float(causal_local_alpha)
        self.causal_local_beta = float(causal_local_beta)
        self.causal_local_expand_pool = bool(causal_local_expand_pool)
        self.causal_prior_mode = str(causal_prior_mode)
        if self.force_fusion not in {"auto", "global", "local", "strict_local", "feature"}:
            raise ValueError(
                f"Unsupported force_fusion={self.force_fusion!r}; "
                "expected 'auto', 'global', 'local', 'strict_local', or 'feature'."
            )
        self.fixed_gate = None if fixed_gate is None else float(fixed_gate)
        if self.fixed_gate is not None and not 0.0 <= self.fixed_gate <= 1.0:
            raise ValueError("fixed_gate must be in [0, 1].")
        self.eval_local_observation_drop_rate = 0.0
        self.eval_local_observation_drop_seed = 0
        self._eval_local_observation_drop_call_count = 0
        self.set_eval_local_observation_drop(
            eval_local_observation_drop_rate,
            eval_local_observation_drop_seed,
        )
        self.point_gate_max = float(point_gate_max)
        if self.point_gate_max <= 0.0:
            raise ValueError("point_gate_max must be positive.")
        self.point_gate_max = min(self.point_gate_max, 1.0)
        if self.causal_context_mode not in {"normal", "zero", "shuffled"}:
            raise ValueError(
                f"Unsupported causal_context_mode={self.causal_context_mode!r}; "
                "expected 'normal', 'zero', or 'shuffled'."
            )
        if self.causal_prior_mode not in {"normal", "zero", "shuffled"}:
            raise ValueError(
                f"Unsupported causal_prior_mode={self.causal_prior_mode!r}; "
                "expected 'normal', 'zero', or 'shuffled'."
            )
        self.num_var_embeddings = max(int(num_vars), 1)
        self.use_node_embedding = num_nodes is not None and int(num_nodes) > 0
        self.num_node_embeddings = max(int(num_nodes or 1), 1)
        self.scale = d_model ** -0.5

        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model),
        )
        self.var_embedding = nn.Embedding(self.num_var_embeddings, d_model)
        if self.use_node_embedding:
            self.node_embedding = nn.Embedding(self.num_node_embeddings, d_model)
        else:
            self.node_embedding = None

        self.same_var_bias = nn.Parameter(torch.tensor(0.5))
        self.time_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.attn_dropout = nn.Dropout(dropout)
        if self.disable_local_branch:
            self.local_prior = None
            self.query_norm = None
            self.local_norm = None
            self.causal_norm = None
            self.local_proj = None
            self.local_refiner = None
            self.utility_gate = None
        else:
            local_prior_mode = "time_weighted" if self.local_fusion_mode == "dual_gate" else self.local_fusion_mode
            self.local_prior = LocalResidualAdapter(
                num_vars=num_vars,
                mode=local_prior_mode,
                k=local_fusion_k,
                causal_local_alpha=self.causal_local_alpha,
                causal_local_beta=self.causal_local_beta,
                causal_local_expand_pool=self.causal_local_expand_pool,
                causal_prior_mode=self.causal_prior_mode,
            )
            self.query_norm = nn.LayerNorm(d_model)
            self.local_norm = nn.LayerNorm(d_model)
            self.causal_norm = nn.LayerNorm(d_model)
            self.local_proj = nn.Linear(2, d_model)
            self.local_refiner = nn.Sequential(
                nn.Linear(d_model * 2 + 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            nn.init.zeros_(self.local_refiner[-1].weight)
            nn.init.zeros_(self.local_refiner[-1].bias)
            if self.local_branch_source == "shared_encoder":
                # Exactly the same encoder representation and query as the global head.
                self.local_prior = None
                self.local_refiner = None
                with torch.random.fork_rng(devices=[]):
                    self.second_output_mlp = nn.Sequential(
                        nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
                    )
            elif self.local_encoder == "attention":
                with torch.random.fork_rng(devices=[]):
                    self.local_prior = RawLocalEncoder(num_vars, d_model, k=local_fusion_k, dropout=dropout)
                self.local_refiner = None
            # Keep the same gate shape in causal and non-causal ablations. The
            # latter receives an all-zero context vector.
            self.utility_gate = nn.Sequential(
                nn.Linear(d_model * 4 + 3 + (4 if self.gate_prediction_features else 0), d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2 if self.separate_utility_head else 1),
            )
            nn.init.zeros_(self.utility_gate[-1].weight)
            nn.init.zeros_(self.utility_gate[-1].bias)
            self.utility_gate[-1].bias.data[0] = float(local_gate_bias)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        if getattr(self, 'local_prior', None) is not None:
            self.local_prior.strict_past = self.recent_strict_past
        self.register_buffer("_gate_stat_sum", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_stat_sq_sum", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_stat_count", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_stat_max", torch.full((1,), -float("inf")), persistent=False)
        self.register_buffer("_gate_stat_min", torch.full((1,), float("inf")), persistent=False)
        self.register_buffer("_gate_local_sum", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_local_sq_sum", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_local_count", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_local_max", torch.full((1,), -float("inf")), persistent=False)
        self.register_buffer("_gate_local_min", torch.full((1,), float("inf")), persistent=False)
        self._query_evidence_enabled = False
        self._query_evidence_cache = []

    def set_eval_local_observation_drop(self, drop_rate=0.0, seed=0):
        drop_rate = float(drop_rate)
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError("eval local observation degradation severity must be in [0, 1].")
        self.eval_local_observation_drop_rate = drop_rate
        self.eval_local_observation_drop_seed = int(seed)
        self._eval_local_observation_drop_call_count = 0

    def _drop_eval_local_history(
        self,
        history_time,
        history_var_id,
        history_batch_id,
        history_value,
        history_node_id=None,
    ):
        drop_rate = float(getattr(self, "eval_local_observation_drop_rate", 0.0))
        if self.training or drop_rate <= 0.0 or history_value is None or history_value.numel() == 0:
            return history_time, history_var_id, history_batch_id, history_value, history_node_id

        num_history = int(history_value.shape[0])
        seed = int(getattr(self, "eval_local_observation_drop_seed", 0))
        seed += int(getattr(self, "_eval_local_observation_drop_call_count", 0))
        self._eval_local_observation_drop_call_count += 1
        generator = torch.Generator(device=history_value.device)
        generator.manual_seed(seed)
        mode = self.eval_local_degradation_mode
        if mode == "stale":
            history_time = history_time - drop_rate * self._local_degradation_time_scale
            return history_time, history_var_id, history_batch_id, history_value, history_node_id
        if mode == "noise":
            noise = torch.randn(num_history, device=history_value.device, generator=generator, dtype=history_value.dtype)
            scale = torch.ones_like(history_value)
            # Noise is relative to each sample/variable's observed standard deviation.
            group_id = self.local_prior._build_group_id(history_var_id, history_node_id)
            for batch_id in history_batch_id.unique():
                for target_group in group_id[history_batch_id == batch_id].unique():
                    group = (history_batch_id == batch_id) & (group_id == target_group)
                    scale[group] = history_value[group].std(unbiased=False).clamp_min(1e-6)
            return history_time, history_var_id, history_batch_id, history_value + drop_rate * scale * noise, history_node_id
        keep_score = torch.rand(num_history, device=history_value.device, generator=generator)
        keep_mask = keep_score >= drop_rate
        if mode == "recent_drop":
            keep_mask = torch.ones_like(keep_mask)
            group_id = self.local_prior._build_group_id(history_var_id, history_node_id)
            for batch_id in history_batch_id.unique():
                for group in group_id[history_batch_id == batch_id].unique():
                    indices = ((history_batch_id == batch_id) & (group_id == group)).nonzero(as_tuple=True)[0]
                    indices = indices[torch.argsort(history_time[indices], descending=True)]
                    count = int(torch.ceil(history_value.new_tensor(drop_rate * indices.numel())).item())
                    keep_mask[indices[:count]] = False

        def _filter(value):
            return None if value is None else value[keep_mask]

        return (
            _filter(history_time),
            _filter(history_var_id),
            _filter(history_batch_id),
            _filter(history_value),
            _filter(history_node_id),
        )

    def reset_local_gate_stats(self):
        self._gate_stat_sum.zero_()
        self._gate_stat_sq_sum.zero_()
        self._gate_stat_count.zero_()
        self._gate_stat_max.fill_(-float("inf"))
        self._gate_stat_min.fill_(float("inf"))
        self._gate_local_sum.zero_()
        self._gate_local_sq_sum.zero_()
        self._gate_local_count.zero_()
        self._gate_local_max.fill_(-float("inf"))
        self._gate_local_min.fill_(float("inf"))

    def _apply_causal_context_mode(self, causal_query_feature):
        if self.causal_context_mode == "zero":
            return torch.zeros_like(causal_query_feature)
        if self.causal_context_mode == "shuffled" and causal_query_feature.shape[0] > 1:
            perm = torch.randperm(causal_query_feature.shape[0], device=causal_query_feature.device)
            return causal_query_feature[perm]
        return causal_query_feature

    def get_local_gate_stats(self):
        gate_count = float(self._gate_stat_count.detach().cpu().item())
        local_count = float(self._gate_local_count.detach().cpu().item())
        if gate_count <= 0.0:
            return None
        gate_mean = float(self._gate_stat_sum.detach().cpu().item() / gate_count)
        gate_sq_mean = float(self._gate_stat_sq_sum.detach().cpu().item() / gate_count)
        gate_std = max(gate_sq_mean - gate_mean * gate_mean, 0.0) ** 0.5
        gate_max = float(self._gate_stat_max.detach().cpu().item())
        gate_min = float(self._gate_stat_min.detach().cpu().item())
        local_max = float(self._gate_local_max.detach().cpu().item()) if local_count > 0.0 else 0.0
        local_min = float(self._gate_local_min.detach().cpu().item()) if local_count > 0.0 else 0.0
        local_mean = (
            float(self._gate_local_sum.detach().cpu().item() / local_count)
            if local_count > 0.0
            else 0.0
        )
        local_sq_mean = (
            float(self._gate_local_sq_sum.detach().cpu().item() / local_count)
            if local_count > 0.0
            else 0.0
        )
        local_std = max(local_sq_mean - local_mean * local_mean, 0.0) ** 0.5
        return {
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "gate_max": gate_max if gate_max != -float("inf") else 0.0,
            "gate_min": gate_min if gate_min != float("inf") else 0.0,
            "point_gate_max": self.point_gate_max,
            "has_local_ratio": float(local_count / gate_count),
            "local_gate_mean": local_mean,
            "local_gate_std": local_std,
            "local_gate_max": local_max if local_max != -float("inf") else 0.0,
            "local_gate_min": local_min if local_min != float("inf") else 0.0,
        }

    def _update_local_gate_stats(self, gate, has_local_history):
        if not self.training or gate.numel() == 0:
            return
        gate_detached = gate.detach()
        self._gate_stat_sum += gate_detached.sum()
        self._gate_stat_sq_sum += gate_detached.pow(2).sum()
        self._gate_stat_count += gate_detached.new_tensor([gate_detached.numel()])
        self._gate_stat_max.copy_(torch.maximum(self._gate_stat_max, gate_detached.max().view(1)))
        self._gate_stat_min.copy_(torch.minimum(self._gate_stat_min, gate_detached.min().view(1)))

        local_mask = has_local_history.detach().to(dtype=torch.bool)
        if local_mask.any():
            local_gate = gate_detached[local_mask]
            self._gate_local_sum += local_gate.sum()
            self._gate_local_sq_sum += local_gate.pow(2).sum()
            self._gate_local_count += local_gate.new_tensor([local_gate.numel()])
            self._gate_local_max.copy_(torch.maximum(self._gate_local_max, local_gate.max().view(1)))
            self._gate_local_min.copy_(torch.minimum(self._gate_local_min, local_gate.min().view(1)))

    def enable_query_evidence_cache(self, enabled=True):
        self._query_evidence_enabled = bool(enabled)
        if enabled:
            self.reset_query_evidence_cache()

    def reset_query_evidence_cache(self):
        self._query_evidence_cache = []

    def pop_query_evidence_cache(self):
        cache = list(getattr(self, "_query_evidence_cache", []))
        self.reset_query_evidence_cache()
        return cache

    def _record_query_evidence(
        self,
        gate,
        gate_logit,
        utility_logit,
        query_time,
        query_var_id,
        query_batch_id,
        has_local_history,
        gate_stats,
        local_stats,
        seq_len,
        global_pred=None,
        local_branch_pred=None,
        fused_pred=None,
        query_node_id=None,
    ):
        if not getattr(self, "_query_evidence_enabled", False) or gate.numel() == 0:
            return
        with torch.no_grad():
            record = {
                "gate": gate.detach().cpu(),
                "gate_logit": gate_logit.detach().cpu(),
                "utility_logit": utility_logit.detach().cpu(),
                "query_time": query_time.detach().cpu(),
                "query_var_id": query_var_id.detach().cpu(),
                "query_batch_id": query_batch_id.detach().cpu(),
                "has_local_history": has_local_history.detach().to(dtype=torch.bool).cpu(),
                "nearest_gap": gate_stats[:, 0].detach().cpu(),
                "recent_gap": gate_stats[:, 0].detach().cpu(),
                "local_density": gate_stats[:, 1].detach().cpu(),
                "valid_history_ratio": gate_stats[:, 2].detach().cpu(),
                "candidate_count": local_stats["candidate_count"].detach().cpu(),
                "same_var_candidate_count": local_stats["same_var_candidate_count"].detach().cpu(),
                "candidate_var_count": local_stats["candidate_var_count"].detach().cpu(),
            }
            record["recent_gap_steps"] = record["recent_gap"] * float(max(int(seq_len) - 1, 1))
            if global_pred is not None:
                record["global_pred"] = global_pred.detach().cpu()
            if local_branch_pred is not None:
                record["local_pred"] = local_branch_pred.detach().cpu()
            if fused_pred is not None:
                record["fused_pred"] = fused_pred.detach().cpu()
            if query_node_id is not None:
                record["query_node_id"] = query_node_id.detach().cpu()
            self._query_evidence_cache.append(record)

    def _validate_var_ids(self, var_id):
        if var_id.numel() == 0:
            return
        min_var_id = int(var_id.min().item())
        max_var_id = int(var_id.max().item())
        if min_var_id < 0 or max_var_id >= self.num_var_embeddings:
            raise ValueError(
                f"Query var_id range [{min_var_id}, {max_var_id}] exceeds configured num_vars={self.num_var_embeddings}."
            )

    def _validate_node_ids(self, node_id):
        if node_id is None or node_id.numel() == 0:
            return
        max_node_id = int(node_id.max().item())
        if max_node_id >= self.num_node_embeddings:
            raise ValueError(
                f"Query node_id {max_node_id} exceeds configured num_nodes={self.num_node_embeddings}."
            )

    def _encode_time(self, time_value, seq_len):
        time_denom = max(int(seq_len) - 1, 1)
        normalized_time = time_value.unsqueeze(-1) / float(time_denom)
        return self.time_encoder(normalized_time)

    def _encode_query(self, query_time, query_var_id, seq_len, query_node_id=None):
        query_var_id = query_var_id.long()
        self._validate_var_ids(query_var_id)
        query_feature = self._encode_time(query_time, seq_len) + self.var_embedding(query_var_id)
        if self.use_node_embedding and query_node_id is not None:
            query_node_id = query_node_id.long().clamp_min(0)
            self._validate_node_ids(query_node_id)
            query_feature = query_feature + self.node_embedding(query_node_id)
        return query_feature

    def _build_token_descriptors(self, token_feature, token_mask, seq_len):
        batch_size, num_vars, n_tokens, _ = token_feature.shape
        device = token_feature.device
        dtype = token_feature.dtype

        token_start = torch.arange(n_tokens, device=device, dtype=dtype) * self.input_token_len
        token_width = (float(seq_len) - token_start).clamp(min=1., max=float(self.input_token_len))
        token_time = token_start + 0.5 * token_width - 0.5
        token_time_feature = self._encode_time(token_time, seq_len).view(1, 1, n_tokens, self.d_model)

        if num_vars > self.num_var_embeddings:
            raise ValueError(
                f"Token feature num_vars={num_vars} exceeds configured num_vars={self.num_var_embeddings}."
            )
        token_var_id = torch.arange(num_vars, device=device, dtype=torch.long)
        token_var_feature = self.var_embedding(token_var_id).view(1, num_vars, 1, self.d_model)

        token_descriptor = token_feature + token_time_feature + token_var_feature
        flat_descriptor = token_descriptor.reshape(batch_size, num_vars * n_tokens, self.d_model)
        flat_mask = token_mask.reshape(batch_size, num_vars * n_tokens)
        flat_var_id = token_var_id.repeat_interleave(n_tokens)
        flat_time = (token_time / float(max(int(seq_len) - 1, 1))).repeat(num_vars)
        return flat_descriptor, flat_mask, flat_var_id, flat_time

    def _predict_global_values(
        self,
        flat_token_feature,
        flat_token_mask,
        flat_token_var_id,
        flat_token_time,
        query_feature,
        query_time_norm,
        query_var_id,
        query_batch_id,
        return_context=False,
        predict=True,
    ):
        pred_value = query_feature.new_zeros(query_feature.shape[0])
        context_value = query_feature.new_zeros(query_feature.shape)
        batch_ids = query_batch_id.unique(sorted=True).detach().cpu().tolist()
        for batch_id in batch_ids:
            local_query_mask = query_batch_id == batch_id
            local_query_idx = local_query_mask.nonzero(as_tuple=False).squeeze(-1)
            if local_query_idx.numel() == 0:
                continue

            valid_token_mask = flat_token_mask[batch_id]
            local_query_feature = query_feature[local_query_idx]
            if valid_token_mask.any():
                local_token_feature = flat_token_feature[batch_id, valid_token_mask]
                attn_logits = torch.matmul(local_query_feature, local_token_feature.transpose(0, 1)) * self.scale

                local_token_var_id = flat_token_var_id[valid_token_mask]
                same_var = query_var_id[local_query_idx].unsqueeze(-1) == local_token_var_id.unsqueeze(0)
                attn_logits = attn_logits + self.same_var_bias * same_var.to(attn_logits.dtype)

                local_token_time = flat_token_time[valid_token_mask]
                rel_time = query_time_norm[local_query_idx].unsqueeze(-1) - local_token_time.unsqueeze(0)
                attn_logits = attn_logits - self.time_bias_scale.abs() * rel_time.abs()

                attn_weight = torch.softmax(attn_logits, dim=-1)
                attn_weight = self.attn_dropout(attn_weight)
                local_context = torch.matmul(attn_weight, local_token_feature)
            else:
                local_context = local_query_feature.new_zeros(local_query_feature.shape)

            if predict:
                pred_value[local_query_idx] = self.output_mlp(
                    torch.cat([local_context, local_query_feature], dim=-1)
                ).squeeze(-1)
            context_value[local_query_idx] = local_context
        if return_context:
            return pred_value, context_value
        return pred_value

    def forward(
        self,
        token_feature,
        token_mask,
        query_time,
        query_var_id,
        query_batch_id,
        seq_len,
        query_node_id=None,
        history_time=None,
        history_var_id=None,
        history_batch_id=None,
        history_value=None,
        history_node_id=None,
        causal_context=None,
        causal_prior=None,
        return_aux=False,
    ):
        flat_token_feature, flat_token_mask, flat_token_var_id, flat_token_time = self._build_token_descriptors(
            token_feature=token_feature,
            token_mask=token_mask,
            seq_len=seq_len,
        )
        query_feature = self._encode_query(
            query_time=query_time,
            query_var_id=query_var_id,
            seq_len=seq_len,
            query_node_id=query_node_id,
        )
        query_time_norm = query_time.to(dtype=token_feature.dtype) / float(max(int(seq_len) - 1, 1))

        if self.disable_local_branch:
            if not hasattr(self, "_debug_global_only_head_printed"):
                print("[DEBUG] point_disable_local_branch = True", flush=True)
                self._debug_global_only_head_printed = True
            global_only_pred = self._predict_global_values(
                flat_token_feature=flat_token_feature,
                flat_token_mask=flat_token_mask,
                flat_token_var_id=flat_token_var_id,
                flat_token_time=flat_token_time,
                query_feature=query_feature,
                query_time_norm=query_time_norm,
                query_var_id=query_var_id,
                query_batch_id=query_batch_id,
            )
            if return_aux:
                zeros = torch.zeros_like(global_only_pred)
                return global_only_pred, {
                    "global_pred": global_only_pred,
                    "local_pred": zeros,
                    "gate": zeros,
                    "gate_logit": zeros,
                    "utility_logit": zeros,
                    "has_local_history": torch.zeros_like(global_only_pred, dtype=torch.bool),
                    "gate_stats": global_only_pred.new_zeros(global_only_pred.shape[0], 3),
                }
            return global_only_pred

        reference_ticket, global_reference = self._eval_local_reference.lookup(
            dict(tokens=token_feature, mask=token_mask, query_time=query_time, query_var_id=query_var_id,
                 query_batch_id=query_batch_id, query_node_id=query_node_id, seq_len=seq_len), token_feature.device,
        )
        if global_reference is None:
            global_pred_value, global_context = self._predict_global_values(
                flat_token_feature=flat_token_feature,
                flat_token_mask=flat_token_mask,
                flat_token_var_id=flat_token_var_id,
                flat_token_time=flat_token_time,
                query_feature=query_feature,
                query_time_norm=query_time_norm,
                query_var_id=query_var_id,
                query_batch_id=query_batch_id,
                return_context=True,
            )
            self._eval_local_reference.store(reference_ticket, (global_pred_value, global_context))
        else:
            global_pred_value, global_context = global_reference

        if self.local_branch_source == "shared_encoder":
            local_pred = self.second_output_mlp(torch.cat((global_context, query_feature), dim=-1)).squeeze(-1)
            # This head never reads raw observations, including their availability.
            count = flat_token_mask.sum(-1).to(token_feature.dtype)[query_batch_id.long()]
            has_local_history = count > 0
            gate_stats = token_feature.new_zeros(query_time.shape[0], 3)
            local_stats = {"candidate_count": count, "same_var_candidate_count": torch.zeros_like(count),
                           "candidate_var_count": torch.zeros_like(count)}
        else:
            self._local_degradation_time_scale = float(max(int(seq_len) - 1, 1))
            history_time, history_var_id, history_batch_id, history_value, history_node_id = self._drop_eval_local_history(
                history_time=history_time, history_var_id=history_var_id, history_batch_id=history_batch_id,
                history_value=history_value, history_node_id=history_node_id,
            )
            local_kwargs = dict(
                query_time=query_time, query_var_id=query_var_id, query_batch_id=query_batch_id, seq_len=seq_len,
                history_time=history_time, history_var_id=history_var_id, history_batch_id=history_batch_id,
                history_value=history_value, query_node_id=query_node_id, history_node_id=history_node_id,
                causal_prior=causal_prior,
            )
            if self.local_encoder == "attention":
                local_kwargs["query_feature"] = query_feature
            local_pred, gate_stats, has_local_history, local_stats = self.local_prior(**local_kwargs)
            if self.local_refiner is not None:
                local_refiner_input = torch.cat(
                    [query_feature, self.local_proj(torch.stack(
                        [local_pred.to(token_feature.dtype), has_local_history.to(token_feature.dtype)], dim=-1
                    )), gate_stats.to(token_feature.dtype)], dim=-1,
                )
                local_pred = local_pred + self.local_refiner(local_refiner_input).squeeze(-1).to(local_pred.dtype)
            local_pred = local_pred * has_local_history.to(local_pred.dtype)

        local_feature = self.local_proj(
            torch.stack(
                [
                    local_pred.to(dtype=token_feature.dtype),
                    has_local_history.to(dtype=token_feature.dtype),
                ],
                dim=-1,
            )
        )
        if causal_context is not None:
            if causal_context.dim() != 3 or causal_context.shape[-1] != self.d_model:
                raise ValueError(
                    f"causal_context must have shape [B, V, {self.d_model}], got {tuple(causal_context.shape)}."
                )
            causal_query_feature = causal_context[
                query_batch_id.long(), query_var_id.long()
            ].to(dtype=token_feature.dtype)
            causal_query_feature = self._apply_causal_context_mode(causal_query_feature)
        else:
            causal_query_feature = torch.zeros_like(query_feature)
        causal_gate_feature = causal_query_feature
        if not self.use_causal_gate_context:
            causal_gate_feature = torch.zeros_like(causal_gate_feature)
        gate_inputs = [
            self.query_norm(query_feature),
            self.local_norm(local_feature),
            self.causal_norm(causal_gate_feature),
            global_context,
            gate_stats.to(dtype=token_feature.dtype) if self.gate_availability_features else torch.zeros_like(gate_stats).to(token_feature.dtype),
        ]
        if self.gate_prediction_features:
            global_gate_pred = global_pred_value.to(dtype=token_feature.dtype)
            local_gate_pred = local_pred.to(dtype=token_feature.dtype)
            prediction_gap = local_gate_pred - global_gate_pred
            gate_inputs.append(torch.stack(
                [global_gate_pred, local_gate_pred, prediction_gap, prediction_gap.abs()],
                dim=-1,
            ))
        gate_input = torch.cat(gate_inputs, dim=-1)
        gate_outputs = self.utility_gate(gate_input)
        if self.separate_utility_head:
            # Both outputs affect fusion, so ordinary and utility modes train
            # the same gate parameters. Only utility mode adds supervision to
            # the second, explicitly interpretable classification logit.
            utility_logit = gate_outputs[:, 1]
            gate_logit = gate_outputs[:, 0] + utility_logit
        else:
            gate_logit = gate_outputs.squeeze(-1)
            utility_logit = gate_logit
        gate = torch.sigmoid(gate_logit)
        if self.point_gate_max < 1.0:
            gate = gate * self.point_gate_max
        if self.gate_mode == "fixed" or self.fixed_gate is not None:
            fixed_value = 0.5 if self.fixed_gate is None else self.fixed_gate
            fixed_value = min(float(fixed_value), self.point_gate_max)
            gate = gate.new_full(gate.shape, fixed_value)
            fixed_gate = gate.clamp(1e-6, 1.0 - 1e-6)
            gate_logit = torch.logit(fixed_gate)
        gate = gate * has_local_history.to(dtype=gate.dtype)
        if not hasattr(self, "_debug_dual_gate_printed"):
            print("[DEBUG] local_fusion_mode =", self.local_fusion_mode, flush=True)
            print("[DEBUG] point_local_fusion_mode =", self.local_fusion_mode, flush=True)
            print("[DEBUG] point_force_fusion =", self.force_fusion, flush=True)
            print("[DEBUG] point_fixed_gate =", self.fixed_gate, flush=True)
            print("[DEBUG] point_gate_max =", self.point_gate_max, flush=True)
            print("[DEBUG] causal_context_mode =", self.causal_context_mode, flush=True)
            print("[DEBUG] causal_context is None:", causal_context is None, flush=True)
            if causal_context is not None:
                print("[DEBUG] causal_context shape:", tuple(causal_context.shape), flush=True)
                causal_norm = causal_context.detach().norm(dim=-1).mean().cpu().item()
                print("[DEBUG] causal_context norm:", causal_norm, flush=True)
            print("[DEBUG] causal_prior shape =", None if causal_prior is None else tuple(causal_prior.shape), flush=True)
            if causal_prior is not None:
                causal_prior_detached = causal_prior.detach()
                causal_prior_mean = causal_prior_detached.mean().cpu().item() if causal_prior_detached.numel() > 0 else 0.0
                causal_prior_std = (
                    causal_prior_detached.std(unbiased=False).cpu().item()
                    if causal_prior_detached.numel() > 0
                    else 0.0
                )
                print("[DEBUG] causal_prior mean/std =", causal_prior_mean, causal_prior_std, flush=True)
            print("[DEBUG] causal_local_alpha =", self.causal_local_alpha, flush=True)
            print("[DEBUG] causal_local_beta =", self.causal_local_beta, flush=True)
            print("[DEBUG] causal_local_expand_pool =", self.causal_local_expand_pool, flush=True)
            print("[DEBUG] causal_prior_mode =", self.causal_prior_mode, flush=True)
            has_local_ratio = has_local_history.float().mean().detach().cpu().item()
            print("[DEBUG] has_local ratio:", has_local_ratio, flush=True)
            candidate_count = local_stats["candidate_count"].detach()
            local_candidate_avg_count = candidate_count.mean().cpu().item() if candidate_count.numel() > 0 else 0.0
            print("[DEBUG] local candidate avg count =", local_candidate_avg_count, flush=True)
            causal_query_norm = causal_query_feature.detach().norm(dim=-1).mean().cpu().item()
            print("[DEBUG] causal_query_feature norm:", causal_query_norm, flush=True)
            gate_detached = gate.detach()
            gate_mean = gate_detached.mean().cpu().item() if gate_detached.numel() > 0 else 0.0
            gate_std = gate_detached.std(unbiased=False).cpu().item() if gate_detached.numel() > 0 else 0.0
            print("[DEBUG] utility_gate mean/std =", gate_mean, gate_std, flush=True)
            if self.causal_local_expand_pool:
                candidate_sum = local_stats["candidate_count"].detach().sum()
                same_var_sum = local_stats["same_var_candidate_count"].detach().sum()
                same_var_candidate_ratio = (
                    (same_var_sum / candidate_sum.clamp_min(1.0)).cpu().item()
                    if candidate_sum.numel() > 0
                    else 0.0
                )
                local_mask = has_local_history.detach().to(dtype=torch.bool)
                if local_mask.any():
                    avg_candidate_vars = local_stats["candidate_var_count"].detach()[local_mask].mean().cpu().item()
                else:
                    avg_candidate_vars = 0.0
                print("[DEBUG] expanded_pool_enabled = True", flush=True)
                print("[DEBUG] same_var_candidate_ratio =", same_var_candidate_ratio, flush=True)
                print("[DEBUG] avg_candidate_vars_per_query =", avg_candidate_vars, flush=True)
            self._debug_dual_gate_printed = True
        self._update_local_gate_stats(gate, has_local_history)
        local_branch_pred_value = local_pred.to(dtype=token_feature.dtype)
        if self.gate_mode == "global" or self.force_fusion == "global":
            pred_value = global_pred_value
        elif self.gate_mode == "local" or self.force_fusion == "local":
            pred_value = torch.where(
                has_local_history,
                local_branch_pred_value,
                global_pred_value,
            )
        elif self.force_fusion == "strict_local":
            pred_value = local_branch_pred_value
        elif self.force_fusion == "feature":
            fused_context = global_context + gate.unsqueeze(-1) * (
                local_feature.to(dtype=global_context.dtype) - global_context
            )
            feature_fused_pred = self.output_mlp(
                torch.cat([fused_context, query_feature], dim=-1)
            ).squeeze(-1)
            pred_value = torch.where(
                has_local_history,
                feature_fused_pred,
                global_pred_value,
            )
        else:
            fused_pred = global_pred_value + gate * (local_branch_pred_value - global_pred_value)
            pred_value = torch.where(
                has_local_history,
                fused_pred,
                global_pred_value,
            )

        self._record_query_evidence(
            gate=gate,
            gate_logit=gate_logit,
            utility_logit=utility_logit,
            query_time=query_time,
            query_var_id=query_var_id,
            query_batch_id=query_batch_id,
            has_local_history=has_local_history,
            gate_stats=gate_stats,
            local_stats=local_stats,
            seq_len=seq_len,
            global_pred=global_pred_value,
            local_branch_pred=local_branch_pred_value,
            fused_pred=pred_value,
            query_node_id=query_node_id,
        )
        if return_aux:
            aux = {
                "global_pred": global_pred_value,
                "local_pred": local_branch_pred_value,
                "gate": gate,
                "gate_logit": gate_logit,
                "utility_logit": utility_logit,
                "has_local_history": has_local_history,
                "gate_stats": gate_stats,
            }
            if self.recent_analysis and not self.training:
                from layers.recent_observations import recent_observations
                *_, diagnostics = recent_observations(
                    query_time, query_var_id, query_batch_id, seq_len, history_time,
                    history_var_id, history_batch_id, history_value, query_node_id,
                    history_node_id, self.recent_k, strict=self.recent_strict_past)
                aux.update(diagnostics)
            return pred_value, aux
        return pred_value

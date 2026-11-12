import torch
from torch import nn

from layers.local_residual_adapter import LocalResidualAdapter


class TimerXLQueryPointHead(nn.Module):
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
        causal_local_max_candidates=128,
        force_fusion="auto",
        fixed_gate=None,
        eval_local_observation_drop_rate=0.0,
        eval_local_observation_drop_seed=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_token_len = input_token_len
        self.local_fusion_mode = str(local_fusion_mode)
        self.force_fusion = str(force_fusion)
        self.causal_context_mode = str(causal_context_mode)
        self.causal_local_alpha = float(causal_local_alpha)
        self.causal_local_beta = float(causal_local_beta)
        self.causal_local_expand_pool = bool(causal_local_expand_pool)
        self.causal_prior_mode = str(causal_prior_mode)
        self.causal_local_max_candidates = int(causal_local_max_candidates)
        if self.force_fusion not in {"auto", "global", "local", "strict_local"}:
            raise ValueError(
                f"Unsupported force_fusion={self.force_fusion!r}; "
                "expected 'auto', 'global', 'local', or 'strict_local'."
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
        local_prior_mode = "time_weighted" if self.local_fusion_mode == "dual_gate" else self.local_fusion_mode
        self.local_prior = LocalResidualAdapter(
            num_vars=num_vars,
            mode=local_prior_mode,
            k=local_fusion_k,
            causal_local_alpha=self.causal_local_alpha,
            causal_local_beta=self.causal_local_beta,
            causal_local_expand_pool=self.causal_local_expand_pool,
            causal_prior_mode=self.causal_prior_mode,
            causal_local_max_candidates=self.causal_local_max_candidates,
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.local_norm = nn.LayerNorm(d_model)
        self.causal_norm = nn.LayerNorm(d_model)
        self.local_proj = nn.Linear(2, d_model)
        self.dual_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.dual_gate[-1].bias, float(dual_gate_bias))
        gate_hidden_dim = max(d_model // 2, 16)
        self.local_gate_mlp = nn.Sequential(
            nn.Linear(d_model + 4, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.local_gate_mlp[-1].weight)
        nn.init.constant_(self.local_gate_mlp[-1].bias, float(local_gate_bias))
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
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
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError("eval local observation drop rate must be in [0, 1).")
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
        try:
            generator = torch.Generator(device=history_value.device)
            generator.manual_seed(seed)
            keep_score = torch.rand(num_history, device=history_value.device, generator=generator)
        except (TypeError, RuntimeError):
            keep_score = torch.rand(num_history, device=history_value.device)
        keep_mask = keep_score >= drop_rate

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

        history_time, history_var_id, history_batch_id, history_value, history_node_id = self._drop_eval_local_history(
            history_time=history_time,
            history_var_id=history_var_id,
            history_batch_id=history_batch_id,
            history_value=history_value,
            history_node_id=history_node_id,
        )

        local_pred, gate_stats, has_local_history, local_stats = self.local_prior(
            query_time=query_time,
            query_var_id=query_var_id,
            query_batch_id=query_batch_id,
            seq_len=seq_len,
            history_time=history_time,
            history_var_id=history_var_id,
            history_batch_id=history_batch_id,
            history_value=history_value,
            query_node_id=query_node_id,
            history_node_id=history_node_id,
            causal_prior=causal_prior,
        )

        use_dual_gate = self.local_fusion_mode in {"dual_gate", "causal_time_weighted"}
        if use_dual_gate:
            if self.local_fusion_mode == "dual_gate":
                if causal_context is None:
                    raise ValueError("dual_gate local fusion requires causal_context with shape [B, V, d_model].")
                if causal_context.dim() != 3 or causal_context.shape[-1] != self.d_model:
                    raise ValueError(
                        f"causal_context must have shape [B, V, {self.d_model}], got {tuple(causal_context.shape)}."
                    )
                causal_query_feature = causal_context[
                    query_batch_id.long(),
                    query_var_id.long(),
                ].to(dtype=token_feature.dtype)
                causal_query_feature = self._apply_causal_context_mode(causal_query_feature)
            else:
                causal_query_feature = torch.zeros_like(query_feature)
            local_feature = self.local_proj(
                torch.stack(
                    [
                        local_pred.to(dtype=token_feature.dtype),
                        has_local_history.to(dtype=token_feature.dtype),
                    ],
                    dim=-1,
                )
            )
            gate_input = torch.cat(
                [
                    self.query_norm(query_feature),
                    self.local_norm(local_feature),
                    self.causal_norm(causal_query_feature),
                ],
                dim=-1,
            )
            gate_logit = self.dual_gate(gate_input).squeeze(-1)
        else:
            gate_input = torch.cat(
                [
                    gate_stats,
                    query_time_norm.unsqueeze(-1),
                    self.var_embedding(query_var_id.long()),
                ],
                dim=-1,
            )
            gate_logit = self.local_gate_mlp(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        if self.point_gate_max < 1.0:
            gate = gate * self.point_gate_max
        if self.fixed_gate is not None:
            gate = gate.new_full(gate.shape, self.fixed_gate)
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
            if use_dual_gate:
                causal_query_norm = causal_query_feature.detach().norm(dim=-1).mean().cpu().item()
                print("[DEBUG] causal_query_feature norm:", causal_query_norm, flush=True)
                gate_detached = gate.detach()
                gate_mean = gate_detached.mean().cpu().item() if gate_detached.numel() > 0 else 0.0
                gate_std = gate_detached.std(unbiased=False).cpu().item() if gate_detached.numel() > 0 else 0.0
                print("[DEBUG] dual_gate mean/std =", gate_mean, gate_std, flush=True)
                print("[DEBUG] dual_gate mean:", gate_mean, flush=True)
                print("[DEBUG] dual_gate std:", gate_std, flush=True)
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
        pred_value = token_feature.new_zeros(query_time.shape[0])
        global_pred_value = token_feature.new_zeros(query_time.shape[0])
        local_branch_pred_value = local_pred.to(dtype=token_feature.dtype)
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

            global_pred = self.output_mlp(
                torch.cat([local_context, local_query_feature], dim=-1)
            ).squeeze(-1)
            global_pred_value[local_query_idx] = global_pred
            if self.force_fusion == "global":
                pred_value[local_query_idx] = global_pred
            elif self.force_fusion == "local":
                pred_value[local_query_idx] = torch.where(
                    has_local_history[local_query_idx],
                    local_pred[local_query_idx],
                    global_pred,
                )
            elif self.force_fusion == "strict_local":
                pred_value[local_query_idx] = local_pred[local_query_idx]
            else:
                local_gate = gate[local_query_idx]
                fused_pred = global_pred + local_gate * (local_pred[local_query_idx] - global_pred)
                pred_value[local_query_idx] = torch.where(
                    has_local_history[local_query_idx],
                    fused_pred,
                    global_pred,
                )

        self._record_query_evidence(
            gate=gate,
            gate_logit=gate_logit,
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

        return pred_value

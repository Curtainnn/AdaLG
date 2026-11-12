import torch
from torch import nn


class LocalResidualAdapter(nn.Module):
    def __init__(
        self,
        num_vars,
        mode="time_weighted",
        k=8,
        causal_local_alpha=0.0,
        causal_local_beta=1.0,
        causal_local_expand_pool=False,
        causal_prior_mode="normal",
        causal_local_max_candidates=128,
    ):
        super().__init__()
        self.num_var_embeddings = max(int(num_vars), 1)
        self.mode = str(mode)
        self.k = max(int(k), 1)
        self.enabled = self.mode != "none"
        self.causal_local_alpha = float(causal_local_alpha)
        self.causal_local_beta = float(causal_local_beta)
        self.causal_local_expand_pool = bool(causal_local_expand_pool)
        self.causal_prior_mode = str(causal_prior_mode)
        self.causal_local_max_candidates = int(causal_local_max_candidates)
        if self.causal_prior_mode not in {"normal", "zero", "shuffled"}:
            raise ValueError(
                f"Unsupported causal_prior_mode={self.causal_prior_mode!r}; "
                "expected 'normal', 'zero', or 'shuffled'."
            )

    def _build_group_id(self, var_id, node_id=None):
        var_id = var_id.long()
        if node_id is None:
            return var_id
        return node_id.long().clamp_min(0) * self.num_var_embeddings + var_id

    def _empty_outputs(self, query_time):
        local_pred = query_time.new_zeros(query_time.shape[0])
        gate_stats = query_time.new_zeros(query_time.shape[0], 3)
        has_local_history = torch.zeros(query_time.shape[0], dtype=torch.bool, device=query_time.device)
        local_stats = {
            "candidate_count": query_time.new_zeros(query_time.shape[0]),
            "same_var_candidate_count": query_time.new_zeros(query_time.shape[0]),
            "candidate_var_count": query_time.new_zeros(query_time.shape[0]),
        }
        return local_pred, gate_stats, has_local_history, local_stats

    def _apply_causal_prior_mode(self, causal_prior):
        if causal_prior is None:
            return None
        if self.causal_prior_mode == "zero":
            return torch.zeros_like(causal_prior)
        if self.causal_prior_mode == "shuffled":
            shuffled = causal_prior
            if shuffled.shape[0] > 1:
                batch_perm = torch.randperm(shuffled.shape[0], device=shuffled.device)
                shuffled = shuffled[batch_perm]
            if shuffled.shape[-1] > 1:
                target_perm = torch.randperm(shuffled.shape[-2], device=shuffled.device)
                source_perm = torch.randperm(shuffled.shape[-1], device=shuffled.device)
                shuffled = shuffled.index_select(-2, target_perm).index_select(-1, source_perm)
            return shuffled
        return causal_prior

    def _forward_causal_time_weighted(
        self,
        query_time,
        query_var_id,
        query_batch_id,
        seq_len,
        history_time=None,
        history_var_id=None,
        history_batch_id=None,
        history_value=None,
        query_node_id=None,
        history_node_id=None,
        causal_prior=None,
    ):
        local_pred, gate_stats, has_local_history, local_stats = self._empty_outputs(query_time)
        if not self.enabled or history_value is None or history_value.numel() == 0 or query_time.numel() == 0:
            return local_pred, gate_stats, has_local_history, local_stats

        if causal_prior is not None and causal_prior.dim() != 3:
            raise ValueError(f"causal_prior must have shape [B, V, V], got {tuple(causal_prior.shape)}.")
        causal_prior = self._apply_causal_prior_mode(causal_prior)

        use_node_group = query_node_id is not None and history_node_id is not None
        time_denom = float(max(int(seq_len) - 1, 1))
        query_time_norm = query_time.to(dtype=history_value.dtype) / time_denom
        history_time_norm = history_time.to(dtype=history_value.dtype) / time_denom
        query_var_id = query_var_id.long()
        history_var_id = history_var_id.long()
        query_batch_id = query_batch_id.long()
        history_batch_id = history_batch_id.long()
        if use_node_group:
            query_node_id = query_node_id.long().clamp_min(0)
            history_node_id = history_node_id.long().clamp_min(0)

        alpha = float(self.causal_local_alpha)
        beta = float(self.causal_local_beta)
        eps = 1e-4

        for query_idx in range(query_time.shape[0]):
            batch_id = query_batch_id[query_idx]
            target_var = query_var_id[query_idx]
            query_t = query_time_norm[query_idx]
            base_mask = (history_batch_id == batch_id) & (history_time_norm <= query_t)
            if use_node_group:
                base_mask = base_mask & (history_node_id == query_node_id[query_idx])

            if self.causal_local_expand_pool:
                same_var_mask = base_mask & (history_var_id == target_var)
                same_var_idx = same_var_mask.nonzero(as_tuple=False).squeeze(-1)
                if same_var_idx.numel() > 0:
                    same_order = torch.argsort(history_time_norm[same_var_idx], descending=True)
                    same_var_idx = same_var_idx[same_order]
                    local_k = min(self.k, int(same_var_idx.numel()))
                    window_start = history_time_norm[same_var_idx[local_k - 1]]
                    candidate_mask = base_mask & (history_time_norm >= window_start)
                    candidate_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
                else:
                    candidate_idx = base_mask.nonzero(as_tuple=False).squeeze(-1)
                    if candidate_idx.numel() > 0:
                        candidate_order = torch.argsort(history_time_norm[candidate_idx], descending=True)
                        candidate_idx = candidate_idx[candidate_order[: self.k]]
            else:
                candidate_mask = base_mask & (history_var_id == target_var)
                candidate_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
                if candidate_idx.numel() > 0:
                    candidate_order = torch.argsort(history_time_norm[candidate_idx], descending=True)
                    candidate_idx = candidate_idx[candidate_order[: self.k]]

            if candidate_idx.numel() == 0:
                continue

            candidate_order = torch.argsort(history_time_norm[candidate_idx], descending=True)
            candidate_idx = candidate_idx[candidate_order]
            # limit number of candidates considered to avoid excessive work
            if int(candidate_idx.numel()) > int(self.causal_local_max_candidates):
                candidate_idx = candidate_idx[: int(self.causal_local_max_candidates)]
            candidate_time = history_time_norm[candidate_idx]
            candidate_value = history_value[candidate_idx]
            candidate_var = history_var_id[candidate_idx]
            time_gap = (query_t - candidate_time).abs()

            score = -torch.log(time_gap + eps)
            if alpha > 0.0 and causal_prior is not None:
                causal_score = causal_prior[
                    batch_id,
                    target_var,
                    candidate_var,
                ].to(dtype=score.dtype)
                same_var = (candidate_var == target_var).to(dtype=score.dtype)
                score = score + alpha * causal_score + beta * same_var

            weight = torch.softmax(score, dim=0)
            local_pred[query_idx] = (weight * candidate_value).sum()
            has_local_history[query_idx] = True

            candidate_count = int(candidate_idx.numel())
            same_var_count = int((candidate_var == target_var).sum().detach().cpu().item())
            candidate_var_count = int(candidate_var.unique().numel())
            nearest_gap = time_gap.min()
            oldest_gap = time_gap.max()
            local_density = torch.log1p(candidate_value.new_tensor(float(candidate_count)) / (oldest_gap + eps))
            valid_history_ratio = min(float(candidate_count), float(self.k)) / float(self.k)

            gate_stats[query_idx, 0] = nearest_gap
            gate_stats[query_idx, 1] = local_density
            gate_stats[query_idx, 2] = valid_history_ratio
            local_stats["candidate_count"][query_idx] = float(candidate_count)
            local_stats["same_var_candidate_count"][query_idx] = float(same_var_count)
            local_stats["candidate_var_count"][query_idx] = float(candidate_var_count)

        return local_pred, gate_stats, has_local_history, local_stats

    def forward(
        self,
        query_time,
        query_var_id,
        query_batch_id,
        seq_len,
        history_time=None,
        history_var_id=None,
        history_batch_id=None,
        history_value=None,
        query_node_id=None,
        history_node_id=None,
        causal_prior=None,
    ):
        local_pred, gate_stats, has_local_history, local_stats = self._empty_outputs(query_time)
        if not self.enabled or history_value is None or history_value.numel() == 0 or query_time.numel() == 0:
            return local_pred, gate_stats, has_local_history, local_stats
        if getattr(self, 'strict_past', False):
            if self.causal_local_expand_pool or self.causal_local_alpha != 0:
                raise ValueError('Strict Recent-K comparison requires same-variable retrieval without causal prior weighting.')
            from layers.recent_observations import recent_observations
            values, gaps, valid, gate_stats, diag = recent_observations(
                query_time, query_var_id, query_batch_id, seq_len, history_time, history_var_id,
                history_batch_id, history_value, query_node_id, history_node_id, self.k, strict=True)
            weight = valid.to(values.dtype) / (gaps + 1e-4)
            local_pred = (values * weight).sum(-1) / weight.sum(-1).clamp_min(1e-6)
            if self.mode == 'last':
                local_pred = values[:, 0]
            count = diag['recent_count']
            return local_pred, gate_stats, diag['has_local_history'], dict(
                candidate_count=count, same_var_candidate_count=count,
                candidate_var_count=diag['has_local_history'].to(values.dtype))
        if self.mode == "causal_time_weighted":
            return self._forward_causal_time_weighted(
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

        use_node_group = query_node_id is not None and history_node_id is not None
        time_denom = float(max(int(seq_len) - 1, 1))
        query_time_norm = query_time.to(dtype=history_value.dtype) / time_denom
        history_time_norm = history_time.to(dtype=history_value.dtype) / time_denom
        query_group_id = self._build_group_id(query_var_id, query_node_id if use_node_group else None)
        history_group_id = self._build_group_id(history_var_id, history_node_id if use_node_group else None)

        batch_ids = query_batch_id.unique(sorted=True).detach().cpu().tolist()
        for batch_id in batch_ids:
            local_query_idx = (query_batch_id == batch_id).nonzero(as_tuple=False).squeeze(-1)
            if local_query_idx.numel() == 0:
                continue

            local_history_idx = (history_batch_id == batch_id).nonzero(as_tuple=False).squeeze(-1)
            if local_history_idx.numel() == 0:
                continue

            local_query_group_id = query_group_id[local_query_idx]
            local_history_group_id = history_group_id[local_history_idx]
            for group_id in local_query_group_id.unique(sorted=True):
                group_query_idx = local_query_idx[local_query_group_id == group_id]
                if group_query_idx.numel() == 0:
                    continue

                group_history_idx = local_history_idx[local_history_group_id == group_id]
                if group_history_idx.numel() == 0:
                    continue

                sort_order = torch.argsort(history_time_norm[group_history_idx])
                sorted_history_idx = group_history_idx[sort_order]
                sorted_history_time = history_time_norm[sorted_history_idx]
                sorted_history_value = history_value[sorted_history_idx]

                group_query_time = query_time_norm[group_query_idx]
                history_pos = torch.searchsorted(sorted_history_time, group_query_time, right=True)
                valid_query_mask = history_pos > 0
                if not valid_query_mask.any():
                    continue

                valid_query_idx = group_query_idx[valid_query_mask]
                valid_history_pos = history_pos[valid_query_mask]
                local_k = min(self.k, int(sorted_history_value.shape[0]))
                gather_offsets = torch.arange(local_k, device=query_time.device, dtype=torch.long)
                valid_count = torch.minimum(valid_history_pos, torch.full_like(valid_history_pos, local_k))
                gather_pos = (valid_history_pos.unsqueeze(-1) - 1 - gather_offsets.unsqueeze(0)).clamp_min(0)
                gather_valid = gather_offsets.unsqueeze(0) < valid_count.unsqueeze(-1)

                candidate_value = sorted_history_value[gather_pos]
                candidate_time = sorted_history_time[gather_pos]
                time_gap = (query_time_norm[valid_query_idx].unsqueeze(-1) - candidate_time).clamp_min(0.0)

                if self.mode == "last":
                    valid_local_pred = candidate_value[:, 0]
                else:
                    weight = (1.0 / (time_gap + 1e-4)) * gather_valid.to(candidate_value.dtype)
                    weight_sum = weight.sum(dim=-1).clamp_min(1e-6)
                    valid_local_pred = (weight * candidate_value).sum(dim=-1) / weight_sum

                has_local_history[valid_query_idx] = True
                local_pred[valid_query_idx] = valid_local_pred

                nearest_gap = time_gap[:, 0]
                oldest_idx = (valid_count - 1).clamp_min(0).unsqueeze(-1)
                oldest_gap = time_gap.gather(1, oldest_idx).squeeze(-1)
                local_density = torch.log1p(valid_count.to(candidate_value.dtype) / (oldest_gap + 1e-4))
                valid_history_ratio = valid_count.to(candidate_value.dtype) / float(self.k)
                gate_stats[valid_query_idx, 0] = nearest_gap
                gate_stats[valid_query_idx, 1] = local_density
                gate_stats[valid_query_idx, 2] = valid_history_ratio
                local_stats["candidate_count"][valid_query_idx] = valid_count.to(
                    dtype=local_stats["candidate_count"].dtype
                )
                local_stats["same_var_candidate_count"][valid_query_idx] = valid_count.to(
                    dtype=local_stats["same_var_candidate_count"].dtype
                )
                local_stats["candidate_var_count"][valid_query_idx] = 1.0

        return local_pred, gate_stats, has_local_history, local_stats


QuerySideLocalPrior = LocalResidualAdapter

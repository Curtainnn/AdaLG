"""A query-conditioned encoder of recent raw same-variable observations."""
import torch
from torch import nn

from layers.local_residual_adapter import LocalResidualAdapter


class RawLocalEncoder(LocalResidualAdapter):
    def __init__(self, num_vars, d_model, k=8, dropout=0.1):
        super().__init__(num_vars=num_vars, mode='time_weighted', k=k)
        self.point_encoder = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.query_proj = nn.Linear(d_model, d_model)
        self.correction = nn.Sequential(
            nn.Linear(d_model * 2 + 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        # Start from the existing time-weighted raw predictor, then learn its correction.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.scale = d_model ** -0.5

    def forward(self, query_feature, **kwargs):
        pred, stats, available, local_stats = super().forward(**kwargs)
        if not available.any():
            return pred, stats, available, local_stats
        qt, qv, qb = (kwargs[key] for key in ('query_time', 'query_var_id', 'query_batch_id'))
        ht, hv, hb, values = (kwargs[key] for key in ('history_time', 'history_var_id', 'history_batch_id', 'history_value'))
        qnode, hnode = kwargs.get('query_node_id'), kwargs.get('history_node_id')
        qgroup = self._build_group_id(qv, qnode if hnode is not None else None)
        hgroup = self._build_group_id(hv, hnode if qnode is not None else None)
        context = query_feature.new_zeros(query_feature.shape)
        for batch_id in qb.unique(sorted=True):
            for group_id in qgroup[qb == batch_id].unique(sorted=True):
                qi = ((qb == batch_id) & (qgroup == group_id) & available).nonzero(as_tuple=True)[0]
                hi = ((hb == batch_id) & (hgroup == group_id)).nonzero(as_tuple=True)[0]
                if qi.numel() == 0 or hi.numel() == 0:
                    continue
                hi = hi[torch.argsort(ht[hi])]
                positions = torch.searchsorted(ht[hi].contiguous(), qt[qi].contiguous(),
                                               right=not getattr(self, 'strict_past', False))
                offsets = torch.arange(min(self.k, hi.numel()), device=qt.device)
                valid = offsets.unsqueeze(0) < positions.unsqueeze(-1)
                indices = hi[(positions.unsqueeze(-1) - 1 - offsets).clamp_min(0)]
                candidate = values[indices]
                gap = (qt[qi, None] - ht[indices]).clamp_min(0) / float(max(int(kwargs['seq_len']) - 1, 1))
                raw = torch.stack((candidate, candidate - candidate[:, :1], torch.log1p(gap)), dim=-1)
                encoded = self.point_encoder(raw.to(query_feature.dtype))
                logits = (encoded * self.query_proj(query_feature[qi]).unsqueeze(1)).sum(-1) * self.scale
                logits = logits - torch.log(gap + 1e-4)
                weights = torch.softmax(logits.masked_fill(~valid, -float('inf')), dim=-1)
                context[qi] = (encoded * weights.unsqueeze(-1)).sum(1)
        correction = self.correction(torch.cat((query_feature, context, stats.to(query_feature.dtype)), dim=-1)).squeeze(-1)
        return pred + correction * available.to(pred.dtype), stats, available, local_stats

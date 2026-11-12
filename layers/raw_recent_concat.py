"""Single prediction head over Global context and a flattened Recent-K MLP."""
import torch
from torch import nn
from layers.adalg_query_point_head import QueryPointHead
from layers.recent_observations import recent_observations


class RawRecentConcatHead(QueryPointHead):
    def __init__(self, *args, concat_hidden=0, **kwargs):
        # Consume the same initialization stream as AdaLG, then remove unused heads.
        # New layers do not perturb initialization of the following Global blocks.
        super().__init__(*args, **kwargs)
        if self.disable_local_branch or self.local_branch_source != 'raw':
            raise ValueError('Raw concat requires a raw Recent-K source.')
        self.k = int(kwargs.get('local_fusion_k', 12))
        target = sum(p.numel() for p in self.parameters())
        for name in ('local_prior', 'query_norm', 'local_norm', 'causal_norm', 'local_proj',
                     'local_refiner', 'utility_gate', 'output_mlp'):
            delattr(self, name)
        shared = sum(p.numel() for p in self.parameters())
        d = self.d_model
        encoder_params = (3 * self.k + 1) * d + (d + 1) * d
        hidden = int(concat_hidden) or max(1, round((target - shared - encoder_params - 1) / (3*d + 5)))
        with torch.random.fork_rng(devices=[]):
            self.recent_mlp = nn.Sequential(nn.Linear(self.k * 3, d), nn.GELU(), nn.Linear(d, d))
            self.concat_head = nn.Sequential(nn.Linear(d * 3 + 3, hidden), nn.GELU(),
                                             nn.Dropout(kwargs.get('dropout', 0.1)), nn.Linear(hidden, 1))
        self.concat_hidden = hidden
        self.parameter_match = dict(reference_head_parameters=target,
                                    concat_head_parameters=sum(p.numel() for p in self.parameters()),
                                    hidden=hidden)

    def forward(self, token_feature, token_mask, query_time, query_var_id, query_batch_id,
                seq_len, query_node_id=None, history_time=None, history_var_id=None,
                history_batch_id=None, history_value=None, history_node_id=None,
                causal_context=None, causal_prior=None, return_aux=False):
        descriptors = self._build_token_descriptors(token_feature, token_mask, seq_len)
        query = self._encode_query(query_time, query_var_id, seq_len, query_node_id)
        _, context = self._predict_global_values(*descriptors, query,
            query_time.to(token_feature.dtype) / max(int(seq_len)-1, 1), query_var_id,
            query_batch_id, return_context=True, predict=False)
        values, gaps, valid, condition, diagnostics = recent_observations(
            query_time, query_var_id, query_batch_id, seq_len, history_time, history_var_id,
            history_batch_id, history_value, query_node_id, history_node_id, self.k, strict=True)
        raw = torch.stack((values, gaps, valid.to(values.dtype)), -1).flatten(1)
        recent = self.recent_mlp(raw.to(query.dtype)) * diagnostics['has_local_history'][:, None]
        pred = self.concat_head(torch.cat((context, query, recent, condition.to(query.dtype)), -1)).squeeze(-1)
        if not return_aux:
            return pred
        # NaN denotes an inapplicable prediction, not a hidden second trained head.
        missing = torch.full_like(pred, float('nan'))
        aux = dict(global_pred=missing, local_pred=missing, gate=torch.zeros_like(pred),
                   gate_logit=torch.zeros_like(pred), utility_logit=torch.zeros_like(pred),
                   has_local_history=diagnostics['has_local_history'], gate_stats=condition)
        aux.update(diagnostics)
        return pred, aux

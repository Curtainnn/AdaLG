"""Deterministic Recent-K retrieval shared by the controlled comparison."""
import torch


def recent_observations(query_time, query_var_id, query_batch_id, seq_len,
                        history_time=None, history_var_id=None, history_batch_id=None,
                        history_value=None, query_node_id=None, history_node_id=None,
                        k=12, strict=True):
    if k < 1:
        raise ValueError('Recent-K must be positive.')
    values = query_time.new_zeros((len(query_time), k))
    gaps = torch.zeros_like(values)
    valid = torch.zeros_like(values, dtype=torch.bool)
    if history_value is not None and history_value.numel():
        for batch in query_batch_id.unique(sorted=True):
            for var in query_var_id[query_batch_id == batch].unique(sorted=True):
                query_group = (query_batch_id == batch) & (query_var_id == var)
                nodes = (query_node_id[query_group].unique(sorted=True)
                         if query_node_id is not None and history_node_id is not None else [None])
                for node in nodes:
                    qm = query_group.clone()
                    hm = (history_batch_id == batch) & (history_var_id == var)
                    if node is not None:
                        qm &= query_node_id == node
                        hm &= history_node_id == node
                    qi, hi = qm.nonzero(as_tuple=True)[0], hm.nonzero(as_tuple=True)[0]
                    if not len(hi):
                        continue
                    hi = hi[torch.argsort(history_time[hi], stable=True)]
                    pos = torch.searchsorted(history_time[hi].contiguous(), query_time[qi].contiguous(),
                                             right=not strict)
                    offset = torch.arange(k, device=query_time.device)
                    mask = offset[None, :] < pos[:, None]
                    indices = hi[(pos[:, None] - 1 - offset[None, :]).clamp_min(0)]
                    values[qi] = torch.where(mask, history_value[indices].to(values.dtype), 0.)
                    gap = (query_time[qi, None] - history_time[indices]) / max(int(seq_len) - 1, 1)
                    gaps[qi] = torch.where(mask, gap.to(gaps.dtype), 0.)
                    valid[qi] = mask
    count = valid.sum(-1)
    available = count > 0
    oldest = gaps.gather(1, (count - 1).clamp_min(0)[:, None]).squeeze(1)
    condition = torch.stack((gaps[:, 0], torch.log1p(count / (oldest + 1e-4)), count / float(k)), -1)
    condition = condition * available[:, None]
    mean = values.sum(-1) / count.clamp_min(1)
    variance = (((values - mean[:, None]) ** 2) * valid).sum(-1) / count.clamp_min(1)
    diagnostics = dict(recent_count=count.to(values.dtype), recent_gap=condition[:, 0],
                       local_dynamics=variance.sqrt(), has_local_history=available)
    return values, gaps, valid, condition, diagnostics

import math

import torch


def build_forecast_query_index(
    query_len,
    seq_len,
    input_token_len,
    output_token_len,
    nonautoregressive,
):
    if nonautoregressive:
        base_time = torch.arange(seq_len, seq_len + query_len, dtype=torch.float32)
        base_slot = torch.arange(query_len, dtype=torch.long)
        return base_time, base_slot

    num_blocks = max(int(math.ceil(query_len / max(output_token_len, 1))), 1)
    time_chunks = []
    slot_chunks = []
    produced = 0
    for block_idx in range(num_blocks):
        block_len = min(output_token_len, query_len - produced)
        if block_len <= 0:
            break
        block_start = input_token_len + block_idx * input_token_len
        block_time = torch.arange(block_start, block_start + block_len, dtype=torch.float32)
        block_slot = torch.arange(
            block_idx * output_token_len,
            block_idx * output_token_len + block_len,
            dtype=torch.long,
        )
        time_chunks.append(block_time)
        slot_chunks.append(block_slot)
        produced += block_len

    if not time_chunks:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )

    return torch.cat(time_chunks, dim=0), torch.cat(slot_chunks, dim=0)


def _ensure_2d_tensor(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError("Expected a 2D [time, var] tensor.")
    return tensor


def build_point_forecast_sample(
    seq_x,
    seq_y,
    input_token_len,
    output_token_len,
    nonautoregressive=False,
    query_mode=None,
    obs_time=None,
    obs_index_time=None,
    obs_var_id=None,
    obs_value=None,
):
    seq_x = _ensure_2d_tensor(seq_x).to(torch.float32)
    seq_y = _ensure_2d_tensor(seq_y).to(torch.float32)

    seq_len, num_vars = seq_x.shape
    query_len = seq_y.shape[0]
    if seq_y.shape[1] != num_vars:
        raise ValueError("seq_x and seq_y must share the same variable dimension.")

    if obs_time is None or obs_var_id is None or obs_value is None:
        obs_mask = torch.isfinite(seq_x)
        obs_index = obs_mask.nonzero(as_tuple=False)
        obs_time = obs_index[:, 0].to(torch.float32)
        obs_index_time = obs_index[:, 0].to(torch.long)
        obs_var_id = obs_index[:, 1].to(torch.long)
        obs_value = seq_x[obs_mask]
    else:
        obs_time = torch.as_tensor(obs_time, dtype=torch.float32)
        if obs_index_time is None:
            obs_index_time = torch.floor(obs_time).to(torch.long)
        else:
            obs_index_time = torch.as_tensor(obs_index_time, dtype=torch.long)
        obs_var_id = torch.as_tensor(obs_var_id, dtype=torch.long)
        obs_value = torch.as_tensor(obs_value, dtype=torch.float32)
        if (
            obs_time.shape[0] != obs_index_time.shape[0]
            or obs_time.shape[0] != obs_var_id.shape[0]
            or obs_time.shape[0] != obs_value.shape[0]
        ):
            raise ValueError("obs_time, obs_index_time, obs_var_id, and obs_value must share the same length.")
    obs_index_time = obs_index_time.clamp(0, seq_len - 1)

    base_query_time, base_query_slot = build_forecast_query_index(
        query_len=query_len,
        seq_len=seq_len,
        input_token_len=input_token_len,
        output_token_len=output_token_len,
        nonautoregressive=nonautoregressive,
    )
    query_time = base_query_time.repeat_interleave(num_vars)
    query_var_id = torch.arange(num_vars, dtype=torch.long).repeat(query_len)
    query_slot = base_query_slot.repeat_interleave(num_vars)

    return {
        "obs_time": obs_time,
        "obs_index_time": obs_index_time,
        "obs_var_id": obs_var_id,
        "obs_value": obs_value,
        "query_time": query_time,
        "query_var_id": query_var_id,
        "query_slot": query_slot,
        "target_value": seq_y.reshape(-1),
        "obs_seq_len": int(seq_len),
        "query_dense_len": int(query_len),
        "num_vars": int(num_vars),
        "query_mode": query_mode or ("future_only" if nonautoregressive else "dense_autoregressive"),
    }


def collate_point_forecast_batch(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    num_vars = samples[0]["num_vars"]
    obs_seq_len = samples[0]["obs_seq_len"]
    query_dense_len = samples[0]["query_dense_len"]
    query_mode = samples[0].get("query_mode", "dense_autoregressive")
    for sample in samples[1:]:
        if sample["num_vars"] != num_vars:
            raise ValueError("All samples in a batch must share the same num_vars.")
        if sample["obs_seq_len"] != obs_seq_len:
            raise ValueError("All samples in a batch must share the same obs_seq_len.")
        if sample["query_dense_len"] != query_dense_len:
            raise ValueError("All samples in a batch must share the same query_dense_len.")
        if sample.get("query_mode", "dense_autoregressive") != query_mode:
            raise ValueError("All samples in a batch must share the same query_mode.")

    obs_time = []
    obs_index_time = []
    obs_var_id = []
    obs_value = []
    obs_batch_id = []
    query_time = []
    query_var_id = []
    query_slot = []
    query_batch_id = []
    target_value = []
    has_obs_node_id = any("obs_node_id" in sample for sample in samples)
    has_query_node_id = any("query_node_id" in sample for sample in samples)
    obs_node_id = []
    query_node_id = []

    for batch_id, sample in enumerate(samples):
        obs_count = sample["obs_time"].shape[0]
        query_count = sample["query_time"].shape[0]

        obs_time.append(sample["obs_time"])
        obs_index_time.append(sample["obs_index_time"].to(torch.long))
        obs_var_id.append(sample["obs_var_id"])
        obs_value.append(sample["obs_value"])
        obs_batch_id.append(torch.full((obs_count,), batch_id, dtype=torch.long))

        query_time.append(sample["query_time"])
        query_var_id.append(sample["query_var_id"])
        query_slot.append(sample["query_slot"])
        query_batch_id.append(torch.full((query_count,), batch_id, dtype=torch.long))
        target_value.append(sample["target_value"])

        if has_obs_node_id:
            if "obs_node_id" in sample:
                obs_node_id.append(sample["obs_node_id"].to(torch.long))
            else:
                obs_node_id.append(torch.full((obs_count,), -1, dtype=torch.long))
        if has_query_node_id:
            if "query_node_id" in sample:
                query_node_id.append(sample["query_node_id"].to(torch.long))
            else:
                query_node_id.append(torch.full((query_count,), -1, dtype=torch.long))

    batch = {
        "obs_time": torch.cat(obs_time, dim=0),
        "obs_index_time": torch.cat(obs_index_time, dim=0),
        "obs_var_id": torch.cat(obs_var_id, dim=0),
        "obs_value": torch.cat(obs_value, dim=0),
        "obs_batch_id": torch.cat(obs_batch_id, dim=0),
        "query_time": torch.cat(query_time, dim=0),
        "query_var_id": torch.cat(query_var_id, dim=0),
        "query_slot": torch.cat(query_slot, dim=0),
        "query_batch_id": torch.cat(query_batch_id, dim=0),
        "target_value": torch.cat(target_value, dim=0),
        "batch_size": len(samples),
        "obs_seq_len": obs_seq_len,
        "query_dense_len": query_dense_len,
        "num_vars": num_vars,
        "query_mode": query_mode,
    }
    if has_obs_node_id:
        batch["obs_node_id"] = torch.cat(obs_node_id, dim=0)
    if has_query_node_id:
        batch["query_node_id"] = torch.cat(query_node_id, dim=0)
    return batch


def build_point_forecast_batch_from_history(
    history,
    input_token_len,
    output_token_len,
    query_len=None,
    history_mask=None,
    nonautoregressive=False,
):
    history = _ensure_2d_tensor(history) if history.dim() == 2 else history
    if history.dim() != 3:
        raise ValueError("Expected a 3D [batch, time, var] history tensor.")

    batch_size, seq_len, num_vars = history.shape
    if history_mask is None:
        history_mask = torch.isfinite(history)
    else:
        history_mask = history_mask.to(dtype=torch.bool, device=history.device)
    safe_history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)

    valid_idx = history_mask.nonzero(as_tuple=False)
    obs_batch_id = valid_idx[:, 0].long()
    obs_time = valid_idx[:, 1].to(torch.float32)
    obs_index_time = valid_idx[:, 1].long()
    obs_var_id = valid_idx[:, 2].long()
    obs_value = safe_history[obs_batch_id, valid_idx[:, 1], obs_var_id]

    query_len = int(query_len or output_token_len)
    if nonautoregressive:
        base_query_time = torch.arange(
            seq_len,
            seq_len + query_len,
            device=history.device,
            dtype=torch.float32,
        )
        base_query_slot = torch.arange(query_len, device=history.device, dtype=torch.long)
    else:
        n_tokens = max(seq_len // input_token_len, 1)
        start_slot = (n_tokens - 1) * output_token_len
        base_query_time = torch.arange(
            seq_len,
            seq_len + query_len,
            device=history.device,
            dtype=torch.float32,
        )
        base_query_slot = torch.arange(
            start_slot,
            start_slot + query_len,
            device=history.device,
            dtype=torch.long,
        )

    query_time = base_query_time.repeat_interleave(num_vars)
    query_var_id = torch.arange(num_vars, device=history.device, dtype=torch.long).repeat(query_len)
    query_slot = base_query_slot.repeat_interleave(num_vars)
    query_batch_id = torch.arange(batch_size, device=history.device, dtype=torch.long).repeat_interleave(query_len * num_vars)

    return {
        "obs_time": obs_time,
        "obs_index_time": obs_index_time,
        "obs_var_id": obs_var_id,
        "obs_value": obs_value,
        "obs_batch_id": obs_batch_id,
        "query_time": query_time.repeat(batch_size),
        "query_var_id": query_var_id.repeat(batch_size),
        "query_slot": query_slot.repeat(batch_size),
        "query_batch_id": query_batch_id,
        "batch_size": batch_size,
        "obs_seq_len": seq_len,
        "query_dense_len": query_len,
        "num_vars": num_vars,
        "query_mode": "rollout_step",
    }


def point_batch_to_dense_history(batch):
    batch_size = int(batch["batch_size"])
    seq_len = int(batch["obs_seq_len"])
    num_vars = int(batch["num_vars"])
    obs_value = batch["obs_value"]
    dense = obs_value.new_zeros(batch_size, seq_len, num_vars)
    mask = torch.zeros(
        batch_size,
        seq_len,
        num_vars,
        dtype=torch.bool,
        device=obs_value.device,
    )

    obs_batch_id = batch["obs_batch_id"].long()
    obs_index_time = batch.get("obs_index_time")
    if obs_index_time is None:
        obs_index_time = batch["obs_time"].long()
    else:
        obs_index_time = obs_index_time.long()
    obs_var_id = batch["obs_var_id"].long()
    dense[obs_batch_id, obs_index_time, obs_var_id] = obs_value
    mask[obs_batch_id, obs_index_time, obs_var_id] = True
    return dense, mask


def query_values_to_dense(values, batch):
    batch_size = int(batch["batch_size"])
    query_dense_len = int(batch["query_dense_len"])
    num_vars = int(batch["num_vars"])
    dense = values.new_zeros(batch_size, query_dense_len, num_vars)
    dense[
        batch["query_batch_id"].long(),
        batch["query_slot"].long(),
        batch["query_var_id"].long(),
    ] = values
    return dense

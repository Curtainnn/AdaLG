"""Freeze the full Global prediction path while training Local and Gate."""
import hashlib

import torch


LOCAL_PREFIXES = (
    'causal_context_proj.',  # used only by the gate, never by Global prediction
    'query_head.local_prior.', 'query_head.local_refiner.',
    'query_head.second_output_mlp.', 'query_head.local_proj.',
    'query_head.query_norm.', 'query_head.local_norm.',
    'query_head.causal_norm.', 'query_head.utility_gate.',
)


def is_local_parameter(name):
    return name.startswith(LOCAL_PREFIXES)


def global_state_hash(state):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(state.items()):
        if is_local_parameter(name):
            continue
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        count += tensor.numel()
    return digest.hexdigest(), count


def prepare_frozen_global(model, source_state, initial_weight=0.1):
    """Strictly restore every Global tensor; leave path-specific Local fresh."""
    from utils.gate_mechanism import reset_gate
    expected = model.state_dict()
    source_keys = {k for k in source_state if not is_local_parameter(k)}
    target_keys = {k for k in expected if not is_local_parameter(k)}
    if source_keys != target_keys:
        raise ValueError(f'Global checkpoint keys differ: missing={sorted(target_keys-source_keys)}, '
                         f'unexpected={sorted(source_keys-target_keys)}')
    for key in target_keys:
        if expected[key].shape != source_state[key].shape:
            raise ValueError(f'Global checkpoint shape differs for {key}: '
                             f'{source_state[key].shape} vs {expected[key].shape}')
    # Validate all keys/shapes before changing the model.
    merged = {k: source_state[k] if k in target_keys else v for k, v in expected.items()}
    model.load_state_dict(merged, strict=True)
    if getattr(model.query_head, 'second_output_mlp', None) is not None:
        # Give the shared-feature control a strong starting point: its second
        # head initially reproduces the already trained Global prediction.
        model.query_head.second_output_mlp.load_state_dict(model.query_head.output_mlp.state_dict(), strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_local_parameter(name))
    if model.query_head.utility_gate is None:
        raise ValueError('Local-path training requires an enabled Local branch and gate.')
    reset_gate(model.query_head.utility_gate, model.query_head.point_gate_max, initial_weight)
    if global_state_hash(model.state_dict()) != global_state_hash(source_state):
        raise RuntimeError('Global tensors changed during Local initialization.')


def train_local_modules(model):
    """Global dropout stays disabled; only Local/Gate modules use train mode."""
    model.eval()
    for prefix in LOCAL_PREFIXES:
        try:
            module = model.get_submodule(prefix.rstrip('.'))
        except AttributeError:
            continue
        if module is not None:
            module.train()

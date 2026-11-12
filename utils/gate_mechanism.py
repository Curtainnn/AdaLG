"""Parameter-matched gate initialization and cap-independent diagnostics."""
import math

import numpy as np
import torch


def initial_gate_logit(weight, cap):
    weight, cap = float(weight), float(cap)
    if not math.isfinite(weight) or not math.isfinite(cap) or not 0 < weight < cap <= 1:
        raise ValueError('Initial gate weight must satisfy 0 < weight < cap <= 1.')
    return math.log(weight / (cap - weight))


def reset_gate(gate, cap, initial_weight=None):
    """Keep the hidden initialization; reset the output to one constant gate."""
    weight = 0.5 * cap if initial_weight is None else float(initial_weight)
    logit = initial_gate_logit(weight, cap)
    with torch.no_grad():
        gate[-1].weight.zero_()
        gate[-1].bias.zero_()
        gate[-1].bias[0] = logit
    return weight


def unrestricted_oracle_error(global_pred, local_pred, target, available):
    """Best convex-mixture error on [0, 1], with exact fallback.

    The same mixture minimizes MSE, MAE, Huber and MAE+MSE pointwise losses.
    Used only after inference for diagnostics; never an inference feature.
    """
    global_pred, local_pred, target = [np.asarray(x, dtype=np.float64)
                                     for x in (global_pred, local_pred, target)]
    direction = local_pred - global_pred
    ratio = np.divide(target - global_pred, direction,
                      out=np.zeros_like(direction), where=direction != 0)
    gate = np.where(np.asarray(available, dtype=bool), np.clip(ratio, 0, 1), 0)
    return global_pred + gate * direction - target

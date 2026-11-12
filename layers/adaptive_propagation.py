"""Deprecated.

The old KNN/neighbor-index propagation implementation was removed from the
CausalODE final version. AdaLG and Timer-XL now use
layers.causal_ode.CausalODEFeaturePropagation.
"""

raise ImportError(
    "adaptive_propagation.py is removed in the CausalODE final version. "
    "Use layers.causal_ode.CausalODEFeaturePropagation instead."
)

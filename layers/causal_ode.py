"""
Constrained Causal ODE propagation blocks.

This module is intentionally different from the old KNN/neighbor-index path.
It does not build neighbor_idx/relation_feature. Instead, it scatters point
features to a dense [B, L, V, H] tensor, generates a dynamic causal matrix
A_t in R^{V x V} with a constrained neural ODE, applies A_t to variable-wise
features, and gathers the propagated features back to observed points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:  # Non-adjoint odeint is faster for small state_dim (32). Adjoint is only worth
    # it when state_dim >> num_params, which is not the case here.
    from torchdiffeq import odeint  # type: ignore
except Exception:
    odeint = None


@dataclass
class CausalODERegularization:
    total: torch.Tensor
    h_loss: torch.Tensor
    l1_loss: torch.Tensor
    diag_loss: torch.Tensor
    laplacian_loss: torch.Tensor


class InputConditionedZ0Encoder(nn.Module):
    """Map sparse observation features to a per-batch ODE initial state.

    Each batch element gets its own z0 by mean-pooling observed point features
    per variable, then projecting to state_dim.  This makes the ODE trajectory
    — and therefore the inferred adjacency A_t — data-dependent rather than
    fixed across all inputs.

    Architecture: per-var mean pool → mean over V → Linear → LayerNorm → Tanh
    The Tanh keeps z0 bounded so the ODE trajectory starts in a controlled range.
    """

    def __init__(self, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.Tanh(),
        )
        # Zero-init the linear weight so z0 starts near zero at the beginning
        # of training (avoids large initial adjacency noise).
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def forward(
        self,
        point_feature: torch.Tensor,   # [N, H]
        batch_idx: torch.Tensor,        # [N]
        var_idx: torch.Tensor,          # [N]
        batch_size: int,
        num_vars: int,
    ) -> torch.Tensor:                  # [B, state_dim]
        H = point_feature.shape[-1]
        b, v = batch_idx.long(), var_idx.long()

        # Per-variable mean pool → [B, V, H]
        var_sum = point_feature.new_zeros(batch_size, num_vars, H)
        var_cnt = point_feature.new_zeros(batch_size, num_vars, 1)
        var_sum.index_put_((b, v), point_feature, accumulate=True)
        var_cnt.index_put_((b, v), point_feature.new_ones(len(point_feature), 1), accumulate=True)
        var_feat = var_sum / var_cnt.clamp_min(1.0)   # [B, V, H]

        # Mask out variables with no observations so their zero-padded features
        # don't dilute the summary.  USHCN has very sparse coverage so this matters.
        var_mask = (var_cnt.squeeze(-1) > 0).to(point_feature.dtype)   # [B, V]
        valid_count = var_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        summary = (var_feat * var_mask.unsqueeze(-1)).sum(dim=1) / valid_count  # [B, H]

        return self.proj(summary)   # [B, state_dim]


class CausalODEFunc(nn.Module):
    """ODE vector field with a DAG-style manifold correction.

    The decoded matrix W(t) is variable-level: W(t) in R^{V x V}. The
    acyclicity penalty h(W)=tr(exp(W∘W))-V follows the NOTEARS/DyCAST style.
    """

    def __init__(
        self,
        state_dim: int,
        num_vars: int,
        hidden_dim: int = 128,
        constraint_strength: float = 0.0,
        correction_clip: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_vars = int(num_vars)
        self.constraint_strength = float(constraint_strength)
        self.correction_clip = float(correction_clip)

        self.drift = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.state_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_vars * self.num_vars),
        )

    def decode_matrix(self, z: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        """Decode state z to adjacency matrix W.

        Args:
            z: [B, state_dim]
            t: scalar time value
        Returns:
            [B, V, V]  — always batched; callers must not assume a squeeze.
        """
        if not torch.is_tensor(t):
            t = z.new_tensor(float(t))
        t_col = t.to(device=z.device, dtype=z.dtype).reshape(1, 1).expand(z.shape[0], 1)
        matrix = self.decoder(torch.cat([z, t_col], dim=-1))
        return matrix.reshape(z.shape[0], self.num_vars, self.num_vars)   # [B, V, V]

    def acyclicity(self, w: torch.Tensor) -> torch.Tensor:
        exp_w = torch.matrix_exp(w * w)
        return exp_w.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - float(self.num_vars)

    def constraint_projection(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.constraint_strength <= 0:
            return torch.zeros_like(z)

        with torch.enable_grad():
            z_req = z if z.requires_grad else z.detach().requires_grad_(True)
            w = self.decode_matrix(z_req, t)
            h = self.acyclicity(w)
            grad_z = torch.autograd.grad(
                h.sum(),
                z_req,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if grad_z is None:
                return torch.zeros_like(z)
            denom = grad_z.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            correction = -(grad_z / denom) * h.reshape(-1, 1)
            correction = torch.tanh(correction / max(self.correction_clip, 1e-6)) * self.correction_clip
            return correction.to(dtype=z.dtype, device=z.device)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Projection is intentionally excluded from the ODE RHS; structural
        # constraints stay in loss-level regularization.
        return self.drift(z)


class DynamicCausalODE(nn.Module):
    """Generate a sequence of dynamic causal matrices A_t."""

    def __init__(
        self,
        num_vars: int,
        state_dim: int = 32,
        hidden_dim: int = 128,
        ode_steps: Optional[int] = 8,
        constraint_strength: float = 0.0,
        h_weight: float = 1e-3,
        l1_weight: float = 1e-4,
        diag_weight: float = 1e-3,
        laplacian_weight: float = 1e-4,
        adjacency_scale: float = 1.0,
    ):
        super().__init__()
        self.num_vars = int(num_vars)
        self.state_dim = int(state_dim)
        self.ode_steps = ode_steps
        self.h_weight = float(h_weight)
        self.l1_weight = float(l1_weight)
        self.diag_weight = float(diag_weight)
        self.laplacian_weight = float(laplacian_weight)
        self.adjacency_scale = float(adjacency_scale)

        self.z0 = nn.Parameter(torch.randn(1, self.state_dim) * 0.02)
        self.func = CausalODEFunc(
            state_dim=self.state_dim,
            num_vars=self.num_vars,
            hidden_dim=int(hidden_dim),
            constraint_strength=float(constraint_strength),
        )
        self.register_buffer("diag_mask", 1.0 - torch.eye(self.num_vars), persistent=False)
        self._last_regularization: Optional[CausalODERegularization] = None
        self._last_adjacency: Optional[torch.Tensor] = None

    def _fallback_rk4(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        states = [z0]
        z = z0
        for i in range(1, t_span.numel()):
            t0 = t_span[i - 1]
            dt = t_span[i] - t_span[i - 1]
            half = dt * 0.5
            k1 = self.func(t0, z)
            k2 = self.func(t0 + half, z + half * k1)
            k3 = self.func(t0 + half, z + half * k2)
            k4 = self.func(t0 + dt, z + dt * k3)
            z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            states.append(z)
        return torch.stack(states, dim=0)   # [T, B, state_dim]

    def integrate(self, z0: torch.Tensor, n_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Integrate the ODE from z0 for n_steps time-points.

        Args:
            z0: [B, state_dim]  — now takes z0 directly (input-conditioned or self.z0)
            n_steps: number of evaluation points in [0, 1]
        Returns:
            states: [n_steps, B, state_dim]
        """
        z0 = z0.to(device=device, dtype=dtype)
        if n_steps <= 1:
            return z0.unsqueeze(0)   # [1, B, state_dim]
        t_span = torch.linspace(0.0, 1.0, n_steps, device=device, dtype=dtype)
        if odeint is not None:
            return odeint(self.func, z0, t_span, method="rk4", atol=1e-5, rtol=1e-4)
        return self._fallback_rk4(z0, t_span)

    def decode_sequence(self, states: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Decode ODE states into adjacency matrices.

        Args:
            states: [T, B, state_dim]
            t_span: [T]
        Returns:
            w_seq: [B, T, V, V]
        """
        matrices = [self.func.decode_matrix(states[i], t_span[i]) for i in range(states.shape[0])]
        # each element: [B, V, V]
        w = torch.stack(matrices, dim=0)      # [T, B, V, V]
        return w.permute(1, 0, 2, 3)          # [B, T, V, V]

    def _update_regularization(self, w_seq: torch.Tensor) -> None:
        # w_seq: [B, T, V, V]
        if w_seq.numel() == 0:
            zero = self.z0.new_tensor(0.0)
            self._last_regularization = CausalODERegularization(zero, zero, zero, zero, zero)
            return
        # matrix_exp inside acyclicity() is expensive: only compute when h_weight > 0.
        # Use batch-mean adjacency to keep the cost at T matrix_exp calls (not B×T).
        if self.h_weight > 0.0:
            w_mean = w_seq.mean(dim=0)   # [T, V, V]
            h_values = torch.stack([self.func.acyclicity(w_mean[t]) for t in range(w_mean.shape[0])])
            h_loss = h_values.pow(2).mean()
        else:
            h_loss = w_seq.new_tensor(0.0)
        l1_loss = w_seq.abs().mean()
        diag_loss = w_seq.diagonal(dim1=-2, dim2=-1).pow(2).mean()
        T = w_seq.shape[1]
        if T > 2:
            lap = w_seq[:, :-2] + w_seq[:, 2:] - 2.0 * w_seq[:, 1:-1]
            laplacian_loss = lap.pow(2).mean()
        else:
            laplacian_loss = w_seq.new_tensor(0.0)
        total = (
            self.h_weight * h_loss
            + self.l1_weight * l1_loss
            + self.diag_weight * diag_loss
            + self.laplacian_weight * laplacian_loss
        )
        self._last_regularization = CausalODERegularization(total, h_loss, l1_loss, diag_loss, laplacian_loss)

    def get_regularization_loss(self) -> torch.Tensor:
        if self._last_regularization is None:
            return self.z0.new_tensor(0.0)
        return self._last_regularization.total

    def resolve_ode_steps(self, seq_len: int) -> int:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len == 1:
            return 1
        requested = int(self.ode_steps) if self.ode_steps is not None else min(seq_len, 8)
        return max(min(requested, seq_len), 2)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        z0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate dynamic adjacency matrices.

        Args:
            seq_len: desired output time length
            z0: [B, state_dim] — input-conditioned initial state from
                InputConditionedZ0Encoder.  If None falls back to self.z0
                (fixed parameter, for debugging/backward compat only).
        Returns:
            a_seq: [B, T, V, V] — per-batch, per-timestep causal adjacency
        """
        if z0 is None:
            z0 = self.z0.to(device=device, dtype=dtype)   # [1, state_dim]

        n_steps = self.resolve_ode_steps(int(seq_len))

        states = self.integrate(z0=z0, n_steps=n_steps, device=device, dtype=dtype)
        # states: [n_steps, B, state_dim]
        t_eval = torch.linspace(0.0, 1.0, n_steps, device=device, dtype=dtype)
        w_seq = self.decode_sequence(states, t_eval)   # [B, n_steps, V, V]
        self._update_regularization(w_seq)

        # [1, 1, V, V] diagonal mask — broadcasts over B and T
        mask = self.diag_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.num_vars <= 1:
            a_small = torch.zeros_like(w_seq)
        else:
            a_small = torch.tanh(w_seq) * mask
        a_small = self.adjacency_scale * a_small          # [B, n_steps, V, V]

        # Linear upsample along the time axis from n_steps -> seq_len
        if n_steps != seq_len:
            B, _, V, _ = a_small.shape
            a_flat = a_small.permute(0, 2, 3, 1).reshape(B, V * V, n_steps)
            a_up = torch.nn.functional.interpolate(
                a_flat, size=seq_len, mode="linear", align_corners=True
            )                                               # [B, V*V, seq_len]
            a_seq = a_up.reshape(B, V, V, seq_len).permute(0, 3, 1, 2)   # [B, T, V, V]
        else:
            a_seq = a_small

        self._last_adjacency = a_seq.detach().mean(0)    # [T, V, V] batch-mean, for viz
        return a_seq                                       # [B, T, V, V]


class CausalODEFeaturePropagation(nn.Module):
    """Apply dynamic CausalODE matrices to observation-point features."""

    def __init__(
        self,
        hidden_dim: int,
        num_vars: int,
        state_dim: int = 32,
        ode_hidden: int = 128,
        constraint_strength: float = 0.0,
        h_weight: float = 1e-3,
        l1_weight: float = 1e-4,
        diag_weight: float = 1e-3,
        laplacian_weight: float = 1e-4,
        adjacency_scale: float = 1.0,
        gate_init: float = -4.0,
        dropout: float = 0.0,
        ode_steps: int = 8,
        causal_prior_window: int = 30,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_vars = int(num_vars)
        self.causal_prior_window = int(causal_prior_window)
        self.graph_ode = DynamicCausalODE(
            num_vars=self.num_vars,
            state_dim=int(state_dim),
            hidden_dim=int(ode_hidden),
            ode_steps=int(ode_steps),
            constraint_strength=float(constraint_strength),
            h_weight=float(h_weight),
            l1_weight=float(l1_weight),
            diag_weight=float(diag_weight),
            laplacian_weight=float(laplacian_weight),
            adjacency_scale=float(adjacency_scale),
        )
        self.update_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        # Input-conditioned z0: encodes observations → per-batch ODE initial state.
        # Zero-initialised so it has no effect at the start of training and
        # gradually learns to condition the ODE on the actual data.
        self.z0_encoder = InputConditionedZ0Encoder(
            hidden_dim=self.hidden_dim,
            state_dim=int(state_dim),
        )

    def _scatter_to_dense(
        self,
        point_feature: torch.Tensor,
        batch_idx: torch.Tensor,
        time_idx: torch.Tensor,
        var_idx: torch.Tensor,
        batch_size: int,
        num_vars: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError("History length must be positive.")
        if not torch.isfinite(time_idx).all() or ((time_idx < 0) | (time_idx > seq_len)).any():
            raise ValueError("Observation timestamps must lie in the closed history interval [0, seq_len].")
        h = point_feature.shape[-1]
        # An inclusive endpoint needs its own cell; do not merge t=L with t=L-1.
        grid_len = max(seq_len, int(time_idx.long().max().item()) + 1) if time_idx.numel() else seq_len
        dense = point_feature.new_zeros(batch_size, grid_len, num_vars, h)
        count = point_feature.new_zeros(batch_size, grid_len, num_vars, 1)
        if point_feature.numel() == 0:
            return dense, count
        t_idx = time_idx.long()
        dense.index_put_((batch_idx.long(), t_idx, var_idx.long()), point_feature, accumulate=True)
        ones = point_feature.new_ones(point_feature.shape[0], 1)
        count.index_put_((batch_idx.long(), t_idx, var_idx.long()), ones, accumulate=True)
        dense = dense / count.clamp_min(1.0)
        return dense, count

    def _apply_source_masked_propagation(
        self,
        dense: torch.Tensor,       # [B, T, V, H]
        adjacency: torch.Tensor,   # [B, T, V_target, V_source]  ← now batch-specific
        source_mask: torch.Tensor, # [B, T, V]
    ) -> torch.Tensor:             # [B, T, V, H]
        # adjacency is already [B, T, V, V] — no expansion needed
        src_mask = source_mask.unsqueeze(2)   # [B, T, 1, V_source]
        weights = adjacency.masked_fill(~src_mask, 0.0)

        denom = weights.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom

        propagated = torch.einsum("btvs,btsh->btvh", weights, dense)
        has_source = source_mask.any(dim=-1, keepdim=True).unsqueeze(-1)
        return torch.where(has_source, propagated, dense)

    def forward(
        self,
        point_feature: torch.Tensor,
        batch_idx: torch.Tensor,
        time_idx: torch.Tensor,
        var_idx: torch.Tensor,
        batch_size: int,
        num_vars: int,
        seq_len: int,
        return_context: bool = False,
    ) -> torch.Tensor:
        if point_feature.numel() == 0:
            if return_context:
                causal_context = point_feature.new_zeros(
                    int(batch_size),
                    int(num_vars),
                    self.hidden_dim,
                )
                causal_prior = point_feature.new_zeros(
                    int(batch_size),
                    int(num_vars),
                    int(num_vars),
                )
                if int(num_vars) > 0:
                    causal_prior.fill_(1.0 / float(int(num_vars)))
                return point_feature, causal_context, causal_prior
            return point_feature
        if int(num_vars) != self.num_vars:
            raise ValueError(f"CausalODEFeaturePropagation was built for num_vars={self.num_vars}, got {num_vars}.")

        dense, count = self._scatter_to_dense(
            point_feature=point_feature,
            batch_idx=batch_idx,
            time_idx=time_idx,
            var_idx=var_idx,
            batch_size=int(batch_size),
            num_vars=int(num_vars),
            seq_len=int(seq_len),
        )
        # z0 = global base (always learned) + per-sample delta (input-conditioned).
        # At init delta_z0 ≈ 0 (zero-init encoder), so the model starts identical
        # to the previous fixed-z0 behaviour and gradually learns data-specific offsets.
        delta_z0 = self.z0_encoder(
            point_feature=point_feature,
            batch_idx=batch_idx,
            var_idx=var_idx,
            batch_size=int(batch_size),
            num_vars=int(num_vars),
        )   # [B, state_dim]
        base_z0 = self.graph_ode.z0.to(
            device=point_feature.device,
            dtype=point_feature.dtype,
        ).expand(int(batch_size), -1)   # [B, state_dim]
        z0 = base_z0 + delta_z0         # [B, state_dim]
        propagation_len = dense.shape[1]
        adjacency = self.graph_ode(
            seq_len=int(seq_len),
            device=point_feature.device,
            dtype=point_feature.dtype,
            z0=z0,
        )   # [B, T, V, V]
        prior_window = max(1, min(int(getattr(self, "causal_prior_window", 30)), adjacency.shape[1]))
        causal_prior = adjacency[:, -prior_window:].mean(dim=1)  # [B, V_target, V_source]
        causal_prior = torch.softmax(causal_prior, dim=-1)
        # Keep the original history-time schedule independent of batch contents.
        # Extend its last matrix to the separate endpoint cell, without resampling
        # adjacency at all earlier times when another sample contains t=L.
        if propagation_len > int(seq_len):
            adjacency = torch.cat([adjacency, adjacency[:, -1:]], dim=1)
        source_mask = count.squeeze(-1) > 0
        propagated = self._apply_source_masked_propagation(
            dense=dense,
            adjacency=adjacency,
            source_mask=source_mask,
        )
        propagated = self.update_proj(propagated)

        gate = torch.sigmoid(self.residual_gate)
        if not hasattr(self, "_debug_causal_stats_printed"):
            print(f"[DEBUG] causal_ode_gate={gate.detach().cpu().item():.6f}", flush=True)
            print(
                f"[DEBUG] causal_ode_steps={self.graph_ode.resolve_ode_steps(int(seq_len))}/{int(seq_len)}",
                flush=True,
            )
            print(f"[DEBUG] observed_source_ratio={source_mask.float().mean().detach().cpu().item():.6f}", flush=True)
            delta_z0_norm = delta_z0.norm(dim=-1).mean().detach().cpu().item()
            z0_norm = z0.norm(dim=-1).mean().detach().cpu().item()
            print(f"[DEBUG] delta_z0_norm={delta_z0_norm:.6f}", flush=True)
            print(f"[DEBUG] z0_norm={z0_norm:.6f}", flush=True)
            self._debug_causal_stats_printed = True

        updated_dense = dense + gate * (propagated - dense)
        updated_dense = torch.where(count > 0, updated_dense, dense)

        gather_t = time_idx.long()
        updated_point = updated_dense[batch_idx.long(), gather_t, var_idx.long()]
        if not return_context:
            return updated_point

        mask = (count > 0).to(propagated.dtype)
        causal_context = (propagated * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return updated_point, causal_context, causal_prior

    def get_regularization_loss(self) -> torch.Tensor:
        return self.graph_ode.get_regularization_loss()

    def get_residual_gate_value(self) -> float:
        return float(torch.sigmoid(self.residual_gate.detach()).cpu().item())

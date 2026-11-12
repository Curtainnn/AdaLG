"""Query-weighted diagnostics for the two predictions from one checkpoint."""
import numpy as np


def complementarity_rows(columns, difficult_quantile=0.75):
    required = ('true', 'pred', 'global_pred', 'local_pred', 'has_local_history', 'gate')
    if not all(key in columns for key in required):
        return []
    values = {key: np.asarray(columns[key]).reshape(-1) for key in required}
    n = len(values['true'])
    if any(len(value) != n for value in values.values()):
        raise ValueError('Branch diagnostics require aligned query arrays.')
    if not 0.0 < difficult_quantile < 1.0:
        raise ValueError('difficult_quantile must be in (0, 1).')
    has_local = values['has_local_history'].astype(bool)
    valid = np.isfinite(values['true']) & np.isfinite(values['pred'])
    valid &= np.isfinite(values['global_pred']) & np.isfinite(values['gate'])
    valid &= ~has_local | np.isfinite(values['local_pred'])
    eg = values['global_pred'] - values['true']
    el = values['local_pred'] - values['true']
    ef = values['pred'] - values['true']
    # Missing-local queries always use global, including the diagnostic oracle.
    local_abs = np.where(has_local, np.abs(el), np.abs(eg))
    local_sq = np.where(has_local, el ** 2, eg ** 2)
    oracle_abs = np.minimum(np.abs(eg), local_abs)
    oracle_sq = np.minimum(eg ** 2, local_sq)
    local_valid = valid & has_local
    threshold = float(np.quantile(np.abs(eg[valid]), difficult_quantile)) if valid.any() else float('nan')
    groups = [('all', valid), ('has_local', local_valid), ('no_local', valid & ~has_local),
              ('global_high_error', valid & (np.abs(eg) >= threshold))]
    if 'query_var_id' in columns:
        var_ids = np.asarray(columns['query_var_id'])
        groups.extend((f'variable_{var}', valid & (var_ids == var)) for var in np.unique(var_ids[valid]))
    if 'forecast_horizon' in columns:
        horizon = np.asarray(columns['forecast_horizon'])
        unique = np.unique(horizon[valid])
        for part, selected in enumerate(np.array_split(unique, min(3, len(unique))), 1):
            groups.append((f'horizon_bin_{part}', valid & np.isin(horizon, selected)))

    def mean(value, selected):
        return float(np.mean(value[selected])) if selected.any() else float('nan')

    rows = []
    for name, selected in groups:
        local_selected = selected & has_local
        g, l = eg[local_selected], el[local_selected]
        corr = (float(np.corrcoef(g, l)[0, 1])
                if len(g) > 1 and np.std(g) > 0 and np.std(l) > 0 else float('nan'))
        rows.append({
            'group': name, 'n_queries': int(selected.sum()), 'n_local_queries': int(local_selected.sum()),
            'global_high_error_threshold': threshold,
            'global_mae': mean(np.abs(eg), selected), 'global_mse': mean(eg ** 2, selected),
            'local_mae_has_local': mean(np.abs(el), local_selected),
            'local_mse_has_local': mean(el ** 2, local_selected),
            'local_with_fallback_mae': mean(local_abs, selected),
            'local_with_fallback_mse': mean(local_sq, selected),
            'fused_mae': mean(np.abs(ef), selected), 'fused_mse': mean(ef ** 2, selected),
            'oracle_selection_mae': mean(oracle_abs, selected), 'oracle_selection_mse': mean(oracle_sq, selected),
            # A convex mixture can beat discrete branch selection; this gap can be negative.
            'fusion_minus_selection_oracle_mae': mean(np.abs(ef) - oracle_abs, selected),
            'local_win_rate': mean((np.abs(el) < np.abs(eg)).astype(float), local_selected),
            'fusion_rescue_rate': mean((np.abs(ef) < np.abs(eg)).astype(float), local_selected),
            'fusion_harm_rate': mean((np.abs(ef) > np.abs(eg)).astype(float), local_selected),
            'mean_local_improvement': mean(np.abs(eg) - np.abs(el), local_selected),
            'mean_fusion_improvement': mean(np.abs(eg) - np.abs(ef), selected),
            'signed_error_correlation': corr, 'mean_gate': mean(values['gate'], selected),
        })
    return rows
